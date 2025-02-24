"""
Copyright to COME Authors, ICLR 2025
built upon on Tent code.
"""
from logging import debug
import os
import time
import argparse
import random
import numpy as np
from pycm import *
from copy import deepcopy
import math
from dataset.selectedRotateImageFolder import prepare_test_data, obtain_train_loader
from utils.metrics import eval_ood, eval_ood_95, get_scores
from utils.utils import get_logger, set_random_seed, generate_mix_data, merge_datasets, generate_balanced_data
from torchvision import datasets as dset
import torch    
import torch.nn.functional as F
import tent, sar, tent_come, sar_come, eata, eata_come
from sam import SAM
import timm
import models.Res as Resnet
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(8)

def get_args():
    parser = argparse.ArgumentParser(description='exps')
    # path
    parser.add_argument('--data', default='/path/to/dataset/Imagenet1K', help='path to dataset')
    parser.add_argument('--data_corruption', default='/path/to/dataset/ImageNet_C', help='path to corruption dataset')
    parser.add_argument('--ood_root', default='/path/to/dataset/', help='path to open-world dataset')
    parser.add_argument('--output', default='/path/to/output/result', help='the output directory of this experiment')
    # dataloader
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--test_batch_size', default=64, type=int, help='batch size for testing')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    # Exp Settings
    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--method', default='Tent', type=str, help='no_adapt, Tent, EATA, SAR, Tent_COME, EATA_COME, SAR_COME')
    parser.add_argument('--model', default='resnet50_bn_torch', type=str, help='resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--model_path', default='path/to/resnet50_bn_torch', type=str, help='path to resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='normal', type=str)
    parser.add_argument('--scoring_function', default='msp', type=str)
    parser.add_argument('--ood_rate', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=1)
    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(1000) * 0.40, type=float, help='the threshold for reliable minimization in SAR')
    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_beta', type=float, default=2000., help='the trade-off between entropy and regularization loss')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon for filtering redundant samples')
    return parser.parse_args()


def get_model(args):
    bs = args.test_batch_size
    if args.model == "vitbase_timm":
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        args.lr = (0.001 / 64) * bs
    elif args.model == "resnet50_bn_torch":
        net = Resnet.__dict__['resnet50'](pretrained=True)
        args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
    else:
        assert False, NotImplementedError
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    return net

def get_adapt_model(net, args):
    if args.method == "no_adapt":
        adapt_model = net.eval()
    elif args.method == "Tent":
        net = tent.configure_model(net)
        params, param_names = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
        adapt_model = tent.Tent(net, optimizer, steps=args.steps)
    elif  args.method=="Tent_COME":
        net = tent_come.configure_model(net)
        params, param_names = tent_come.collect_params(net)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
        adapt_model = tent_come.Tent_COME(net, optimizer, steps=args.steps,args=args)
             
    elif args.method == 'SAR':
        net = sar.configure_model(net)
        params, param_names = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
        
    elif args.method =='SAR_COME':
        net = sar_come.configure_model(net)
        params, param_names = sar_come.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
        adapt_model = sar_come.SAR_COME(net, optimizer, margin_e0=args.sar_margin_e0)

    elif args.method == "EATA":
            # compute fisher informatrix
        temp = args.corruption
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)
        args.corruption = temp

        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        #logger.info(param_names)
        # fishers = None
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = net(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_beta, e_margin=args.e_margin, d_margin=args.d_margin)

    elif args.method == "EATA_COME":
            # compute fisher informatrix
        temp = args.corruption
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)
        args.corruption = temp
        net = eata_come.configure_model(net)
        params, param_names = eata_come.collect_params(net)
        #logger.info(param_names)
        # fishers = None
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = net(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = eata_come.EATA_COME(net, optimizer, fishers, args.fisher_beta, e_margin=args.e_margin, d_margin=args.d_margin)
    else:
        assert False, NotImplementedError

    return adapt_model

def create_ood_dataset(ood_root):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_pipeline = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    datasets_dict = {
        "Ninco": dset.ImageFolder(root=os.path.join(ood_root, "NINCO/NINCO_OOD_classes"), transform=transform_pipeline),
        "iNaturalist": dset.ImageFolder(root=os.path.join(ood_root, "iNaturalist/train_val_images"), transform=transform_pipeline),
        "SSB_Hard": dset.ImageFolder(root=os.path.join(ood_root, "ssb_hard_3"), transform=transform_pipeline),
        "Texture": dset.ImageFolder(root=os.path.join(ood_root, "dtd/images"), transform=transform_pipeline),
        "Openimage_O": dset.ImageFolder(root=os.path.join(ood_root, "openimage_o_3"), transform=transform_pipeline)
    }
    
    OOD_dataset = merge_datasets(list(datasets_dict.values()))
    
    return OOD_dataset, datasets_dict


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.output): 
        os.makedirs(args.output, exist_ok=True)

    args.logger_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-{}-{}-level{}-seed{}-ood_rate{}-{}.txt".format(args.method, args.model, args.level, args.seed,args.ood_rate,args.exp_type)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 
     
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    
    if args.exp_type == 'normal':
        cpt_name,accs, fprs, aurocs = [], [], [],[]
        for corrupt in common_corruptions:
            net = get_model(args)
            adapt_model = get_adapt_model(net, args)
            args.corruption = corrupt
            
            ID_dataset, _ = prepare_test_data(args)
            ID_dataset.switch_mode(True, False)
            mixed_data = generate_mix_data(ID_dataset,[],0)
            mixed_loader = DataLoader(mixed_data, batch_size = args.test_batch_size, shuffle=True,
                            num_workers = args.workers, pin_memory = True)
            in_score, out_score, acc = get_scores(args,adapt_model, mixed_loader)
            fpr, auroc, aupr = eval_ood(in_score, out_score)
            cpt_name.append(corrupt) 
            accs.append(acc)        
            fprs.append(fpr)       
            aurocs.append(auroc)     
            logger.info(f"Result under {corrupt}. Accuracy: {acc:.5f}, fpr: {fpr:.5f}, AUROC: {auroc:.5f}")
        logger.info("\n")
        args_str = f"method: {args.method}, level: {args.level}, exp_type: {args.exp_type}, steps: {args.steps}, scoring_function: {args.scoring_function}, model: {args.model}, ood_rate: {args.ood_rate}, seed: {args.seed}"
        logger.info(args_str)
        logger.info(f"Completed corruptions: {cpt_name}")
        logger.info(f"Accuracies: {accs}")
        logger.info(f"FPRs: {fprs}")
        logger.info(f"AUROCs: {aurocs}") 
    elif args.exp_type == 'open-world':
        logger.info(args)
        args.corruption = 'gaussian_noise'
        ID_dataset, _ = prepare_test_data(args)

        ID_dataset.switch_mode(True, False)
        _, individual_datasets = create_ood_dataset(args.ood_root)
        OOD_datasets = ['None','Ninco', 'iNaturalist', 'SSB_Hard', 'Texture','Openimage_O']
        ood_names,accs, fprs, aurocs, thresholds95s = [], [], [],[],[]
        for ood_name in OOD_datasets:
            net = get_model(args)
            adapt_model = get_adapt_model(net, args)
            if ood_name == 'None':
                mixed_data = generate_balanced_data(ID_dataset,[],0)
            else: 
                mixed_data = generate_balanced_data(ID_dataset,individual_datasets[ood_name],args.ood_rate)

            mixed_loader = DataLoader(mixed_data, batch_size = args.test_batch_size, shuffle=True,
                                num_workers = args.workers, pin_memory = True)
            in_score, out_score, acc = get_scores(args,adapt_model, mixed_loader)

            fpr, auroc, aupr, thresholds95 = eval_ood_95(in_score, out_score)
            ood_names.append(ood_name)  
            accs.append(acc)          
            fprs.append(fpr)         
            aurocs.append(auroc)      
            thresholds95s.append(thresholds95)
            logger.info(f"Result under {ood_name}. Accuracy: {acc:.5f}, fpr: {fpr:.5f}, AUROC: {auroc:.5f},thresholds95: {thresholds95}")
        logger.info("\n")
        args_str = f"method: {args.method}, level: {args.level}, exp_type: {args.exp_type}, steps: {args.steps}, scoring_function: {args.scoring_function}, model: {args.model}, ood_rate: {args.ood_rate}, seed: {args.seed}"
        logger.info(args_str)
        logger.info(f"Completed: {ood_names}")
        logger.info(f"Accuracies: {accs}")
        logger.info(f"FPRs: {fprs}")
        logger.info(f"thresholds95: {thresholds95s}")
        logger.info(f"AUROCs: {aurocs}")  
    elif args.exp_type == 'imblanced':
        cpt_name,accs, fprs, aurocs = [], [], [],[]
        for corrupt in common_corruptions:
            net = get_model(args)
            adapt_model = get_adapt_model(net, args)
            args.corruption = corrupt
            ID_dataset, _ = prepare_test_data(args)

            ID_dataset.switch_mode(True, False)
            mixed_data = generate_mix_data(ID_dataset,[],0)
            mixed_loader = DataLoader(mixed_data, batch_size = args.test_batch_size, shuffle=False,
                                num_workers = args.workers, pin_memory = True)
            in_score, out_score, acc = get_scores(args,adapt_model, mixed_loader)
            
            fpr, auroc, aupr = eval_ood(in_score, out_score)
            cpt_name.append(corrupt)  
            accs.append(acc)          
            fprs.append(fpr)         
            aurocs.append(auroc)      
            logger.info(f"Result under {corrupt}. Accuracy: {acc:.5f}, fpr: {fpr:.5f}, AUROC: {auroc:.5f}")
        logger.info("\n")
        args_str = f"method: {args.method}, level: {args.level}, exp_type: {args.exp_type}, steps: {args.steps}, scoring_function: {args.scoring_function}, model: {args.model}, ood_rate: {args.ood_rate}, seed: {args.seed}"
        logger.info(args_str)
        logger.info(f"Completed: {cpt_name}")
        logger.info(f"Accuracies: {accs}")
        logger.info(f"FPRs: {fprs}")
        logger.info(f"AUROCs: {aurocs}") 
    elif args.exp_type == 'mix-shift':
        net = get_model(args)
        adapt_model = get_adapt_model(net, args)
        ID_datasets = []
        for corrupt in common_corruptions:
            args.corruption = corrupt
            ID_dataset, _ = prepare_test_data(args)
            ID_dataset.switch_mode(True, False)
            ID_datasets.append(ID_dataset)
        ID_dataset = ConcatDataset(ID_datasets)
        mixed_data = generate_mix_data(ID_dataset,[],0)
        mixed_loader = DataLoader(mixed_data, batch_size = args.test_batch_size, shuffle=True,
                            num_workers = args.workers, pin_memory = True)
        in_score, out_score, acc = get_scores(args,adapt_model, mixed_loader)
        fpr, auroc, aupr = eval_ood(in_score, out_score)

        logger.info("\n")
        args_str = f"method: {args.method}, level: {args.level}, exp_type: {args.exp_type}, steps: {args.steps}, scoring_function: {args.scoring_function}, model: {args.model}, ood_rate: {args.ood_rate}, seed: {args.seed}"
        logger.info(args_str)
        logger.info(f"Completed: mix_shift")
        logger.info(f"Accuracies: {acc}")
        logger.info(f"FPRs: {fpr}")
        logger.info(f"AUROCs: {auroc}") 

    elif args.exp_type == 'life-long':
        cpt_name,accs, fprs, aurocs = [], [], [],[]
        in_scores,out_scores=[],[]
        net = get_model(args)
        adapt_model = get_adapt_model(net, args)
        for corrupt in common_corruptions:
            args.corruption = corrupt
            ID_dataset, _ = prepare_test_data(args)
            ID_dataset.switch_mode(True, False)
            mixed_data = generate_mix_data(ID_dataset,[],0)
            mixed_loader = DataLoader(mixed_data, batch_size = args.test_batch_size, shuffle=True,
                                num_workers = args.workers, pin_memory = True)
            in_score, out_score, acc = get_scores(args,adapt_model, mixed_loader)
            in_scores.extend(in_score)
            out_scores.extend(out_score)
            fpr, auroc, aupr = eval_ood(in_scores, out_scores)
            cpt_name.append(corrupt)  
            accs.append(acc)          
            fprs.append(fpr)          
            aurocs.append(auroc)      
            logger.info(f"Result under {corrupt}. Accuracy: {acc:.5f}, fpr: {fpr:.5f}, AUROC: {auroc:.5f}")
        logger.info("\n")
        args_str = f"method: {args.method}, level: {args.level}, exp_type: {args.exp_type}, steps: {args.steps}, scoring_function: {args.scoring_function}, model: {args.model}, ood_rate: {args.ood_rate}, seed: {args.seed}"
        logger.info(args_str)
        logger.info(f"Completed: {cpt_name}")
        logger.info(f"Accuracies: {accs}")
        logger.info(f"FPRs: {fprs}")
        logger.info(f"AUROCs: {aurocs}")      
    else:
        assert False, NotImplementedError