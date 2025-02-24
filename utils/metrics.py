import logging
import time
import random
import numpy as np
import sys
import os
import argparse
import torch
import csv
import foolbox as fb
import torch.nn.functional as F
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score, auc, precision_recall_curve
from utils.third_party import _augmix_aug as tr_transforms
from tqdm import tqdm

def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    '''
    compute fnr at 05
    '''
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    
    if len(set(y_true)) < 2:
        print("Warning: Only one class present in y_true. Skipping DET curve calculation.")
        return 1.0  

    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]
    thresholds95 = thresholds[idx]
    
    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    
    return fnr_at_fpr_cutoff, thresholds95




def compute_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc



def compute_aupr(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)

    return aupr


def eval_ood(in_scores, out_scores):
    fpr,_ = compute_fnr(out_scores, in_scores)
    auroc = compute_auroc(out_scores, in_scores)
    aupr = compute_aupr(out_scores, in_scores)

    return fpr, auroc, aupr

def eval_ood_95(in_scores, out_scores):
    fpr,fprs95 = compute_fnr(out_scores, in_scores)
    auroc = compute_auroc(out_scores, in_scores)
    aupr = compute_aupr(out_scores, in_scores)

    return fpr, auroc, aupr,fprs95
def dirichlet_probability(x):
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    """Entropy of softmax distribution from logits."""
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = 1000 / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    

    return brief

def ood_metric(output, scoring_function='dirichlet'):
    to_np = lambda x: x.data.cpu().numpy()
    
    if scoring_function == 'entropy':
        score = to_np(output.mean(1) - torch.logsumexp(output, dim=1))
    elif scoring_function == 'energy':
        score = to_np(-torch.logsumexp(output, dim=1))
    elif scoring_function == 'msp':
        score = -np.max(to_np(F.softmax(output, dim=1)), axis=1)
    elif scoring_function == 'dirichlet':
        score = -np.max(to_np(dirichlet_probability(output).cpu()), axis=1)
    else:
        raise ValueError(f"Unknown scoring function: {scoring_function}")

    if not isinstance(score, np.ndarray):
        score = np.array([score])
    
    return score
def get_scores(args,net, test_loader):
    _in_score, _out_score = [], []
    correct = 0
    total = 0

    loader = test_loader

    for batch in tqdm(loader, total=len(loader), disable=False):
        test_set = batch
        sample, ood_label = test_set[0], test_set[1]
            
        test_data, target, ood_label = sample[0].cuda(), sample[1].cuda(), ood_label.cuda()

        output = net(test_data)

        score = ood_metric(output, scoring_function=args.scoring_function)
            
        
        mask_in = (ood_label == 0)
        mask_in = mask_in.cpu()
            
        if mask_in.any():
            predictions = output.max(1)[1] 
            mask_right = ((predictions == target).cpu() & mask_in).cpu()
            mask_wrong = ((predictions != target).cpu() & mask_in).cpu()
            _in_score.extend(score[mask_right].tolist())
            
            correct += (predictions[mask_in]==target[mask_in]).sum().item()
            total += mask_in.sum().item()
            
        
        mask_out = ood_label == 1
        mask_out = mask_out.cpu()
        if mask_out.any():
            _out_score.extend(score[mask_out].tolist())

        if mask_wrong.any():
            _out_score.extend(score[mask_wrong].tolist())

    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy
def get_scores_test(args,net, test_loader):
    _in_score, _out_score = [], []
    correct = 0
    total = 0

    loader = test_loader

    for batch in tqdm(loader, total=len(loader), disable=False):
        test_set = batch
        sample, ood_label = test_set[0], test_set[1]
            
        test_data, target, ood_label = sample[0].cuda(), sample[1].cuda(), ood_label.cuda()

        output = net(test_data)

        score = ood_metric(output, scoring_function=args.scoring_function)
            
        
        mask_in = (ood_label == 0)
        mask_in = mask_in.cpu()
            
        if mask_in.any():
            predictions = output.max(1)[1] 
            mask_right = ((predictions == target).cpu() & mask_in).cpu()
            mask_wrong = ((predictions != target).cpu() & mask_in).cpu()
            _in_score.extend(score[mask_right].tolist())
            
            correct += (predictions[mask_in]==target[mask_in]).sum().item()
            total += mask_in.sum().item()
            
        
        mask_out = ood_label == 1
        mask_out = mask_out.cpu()
        if mask_out.any():
            _out_score.extend(score[mask_out].tolist())

        if mask_wrong.any():
            _out_score.extend(score[mask_wrong].tolist())

    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy
def get_scores_memo(args, net, test_loader):
    _in_score, _out_score = [], []
    correct = 0
    total = 0

    loader = test_loader
    with tqdm(total=len(loader), disable=True) as _tqdm:  
        _tqdm.set_description('Processing batches')
        for batch in loader:
            test_set = batch
            sample, ood_label = test_set[0], test_set[1]
                
            test_data, target = sample
            ood_label = ood_label
            
            output = net(test_data)

            score = ood_metric(output, scoring_function=args.scoring_function)
                
            
            mask_in = (ood_label == 0).item()
            if mask_in:
                predictions = output.max(1)[1]
                mask_right = (predictions == target).cpu().item()
                mask_wrong = (predictions != target).cpu().item()
                if mask_right:
                    _in_score.extend(score.tolist())
                elif mask_wrong:
                    _out_score.extend(score.tolist())
                correct += mask_right
                total += 1
            
            mask_out = (ood_label == 1).item()
            if mask_out:
                _out_score.extend(score.tolist())

            
            accuracy = correct / total if total > 0 else 0
            _tqdm.set_postfix(accuracy='{:.4f}'.format(accuracy))
            _tqdm.update(1)  

    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy

def dirichlet_entropy(x):
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    
    """Entropy of softmax distribution from logits."""
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = 1000 / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    return -(probability * torch.log(probability)).sum(1) 
def dirichlet_uncertainty(x):
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    
    """Entropy of softmax distribution from logits."""
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = 1000 / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    return uncertainty
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def get_scores_pot(args, net, test_loader):
    save_path="pot/"+args.method + "_" + args.corruption + "_level"+str(args.level)+".csv"
    _in_score, _out_score = [], []
    correct = 0
    total = 0
    batch_accuracies = []
    batch_fprs = []
    batch_softmax_max = []
    batch_uncertainty = []
    softmax_Entropys = []
    dirichlet_Entropys = []

    loader = test_loader

    for batch in tqdm(loader, total=len(loader), disable=True):
        test_set = batch
        sample, ood_label = test_set[0], test_set[1]
            
        test_data, target, ood_label = sample[0].cuda(), sample[1].cuda(), ood_label.cuda()

        output = net(test_data)
        uncertainty = dirichlet_uncertainty(output).mean(0).item()
        softmax_Entropy = softmax_entropy(output).mean(0).item()
        dirichlet_Entropy = dirichlet_entropy(output).mean(0).item()

        score = ood_metric(output, scoring_function=args.scoring_function)
            
        
        mask_in = (ood_label == 0)
        mask_in = mask_in.cpu()
            
        if mask_in.any():
            predictions = output.max(1)[1] 
            mask_right = ((predictions == target).cpu() & mask_in).cpu()
            mask_wrong = ((predictions != target).cpu() & mask_in).cpu()
            _in_score.extend(score[mask_right].tolist())
            correct += (predictions[mask_in]==target[mask_in]).sum().item()
            total += mask_in.sum().item()

            
            accuracy = correct / total if total > 0 else 0
            softmax_max = torch.softmax(output, dim=1).max(1)[0].mean().item()
            

        
        mask_out = ood_label == 1
        mask_out = mask_out.cpu()
        if mask_out.any():
            _out_score.extend(score[mask_out].tolist())

        if mask_wrong.any():
            _out_score.extend(score[mask_wrong].tolist())
        fpr = compute_fnr(_out_score, _in_score)
        batch_fprs.append(fpr)
        batch_accuracies.append(accuracy)
        batch_softmax_max.append(softmax_max)
        softmax_Entropys.append(softmax_Entropy)
        dirichlet_Entropys.append(dirichlet_Entropy)
        batch_uncertainty.append(uncertainty)

    
    with open(save_path, 'w', newline='') as csvfile:
        
        fieldnames = ['batch', 'accuracy','fpr', 'softmax_max','softmax_Entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, (acc,fpr, softmax,se,de,u) in enumerate(zip(batch_accuracies,batch_fprs, batch_softmax_max,softmax_Entropys,dirichlet_Entropys,batch_uncertainty)):
            
            writer.writerow({'batch': i + 1, 'accuracy': acc, 'fpr':fpr,'softmax_max': softmax,'softmax_Entropy':se})
    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy


def get_scores_pot_u(args, net, test_loader):
    save_path = "pot_u/" + args.method + "_" + args.corruption + "_level" + str(args.level) + ".csv"
    _in_score, _out_score = [], []
    correct = 0
    total = 0

    
    sample_info = []

    loader = test_loader
    sample_id = 0  

    for batch in tqdm(loader, total=len(loader), disable=True):
        test_set = batch
        sample, ood_label = test_set[0], test_set[1]
            
        test_data, target, ood_label = sample[0].cuda(), sample[1].cuda(), ood_label.cuda()

        output= net(test_data)
        uncertainty = dirichlet_uncertainty(output)

        score = ood_metric(output, scoring_function=args.scoring_function)
        predictions = output.max(1)[1]
        softmax_scores = torch.softmax(output, dim=1).max(1)[0]
        
        mask_in = (ood_label == 0).cpu()
        mask_out = (ood_label == 1).cpu()
        
        mask_right = ((predictions == target).cpu() & mask_in)
        mask_wrong = ((predictions != target).cpu() & mask_in)
        
        
        
        for i in range(len(test_data)):
            if mask_in[i] and mask_right[i]:
                flag = 0  
            elif mask_in[i] and mask_wrong[i]:
                flag = 1  
            elif mask_out[i]:
                flag = 2  
            else:
                continue
            
            entry = {
                'id': sample_id,
                'softmax_max': softmax_scores[i].item(),
                'flag': flag
            }
            
            
            if 'dirichlet' in args.method.lower():
                entry['u'] = uncertainty[i].item()
            
            sample_info.append(entry)
            sample_id += 1

        if mask_in.any():
            _in_score.extend(score[mask_right].tolist())
            correct += (predictions[mask_in]==target[mask_in]).sum().item()
            total += mask_in.sum().item()
            
        if mask_out.any():
            _out_score.extend(score[mask_out].tolist())

        if mask_wrong.any():
            _out_score.extend(score[mask_wrong].tolist())

    
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'softmax_max', 'u', 'flag'] if 'dirichlet' in args.method.lower() else ['id', 'softmax_max', 'flag']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sample_info:
            writer.writerow(entry)
    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy



def normalize_tensor(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
    std = torch.tensor(std).view(1, 3, 1, 1).cuda()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def merge_tensors(tensor1, tensor2, p):
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    mask = torch.rand(tensor1.shape[0], 1, 1, 1).to(tensor1.device) > p
    mask = mask.expand_as(tensor1)
    merged_tensor = torch.where(mask, tensor1, tensor2)
    return merged_tensor

def get_scores_adversial(net, test_loader, fmodel, attack, args):
    _in_score, _out_score = [], []
    correct = 0
    total = 0

    loader = test_loader

    count = 0

    
    for batch in tqdm(loader, desc="Processing"):
        count += 1
        
        test_set = batch
        sample, ood_label = test_set[0], test_set[1]
            
        test_data, target, ood_label = sample[0].cuda(), sample[1].cuda(), ood_label.cuda()

        fpredict = fmodel(test_data).max(1)[1]

        
        _, advs, _ = attack(fmodel, test_data, fpredict, epsilons=[args.epsilon])
        advs = advs[0]

        mix_data = merge_tensors(test_data, advs, p=args.ad_rate)
        

        mix_data = normalize_tensor(mix_data, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        output = net(mix_data)

        score = ood_metric(output, scoring_function=args.scoring_function)
            
        
        mask_in = (ood_label == 0)
        mask_in = mask_in.cpu()
            
        if mask_in.any():
            predictions = output.max(1)[1] 
            mask_right = ((predictions == target).cpu() & mask_in).cpu()
            mask_wrong = ((predictions != target).cpu() & mask_in).cpu()
            _in_score.extend(score[mask_right].tolist())
            correct += (predictions[mask_in]==target[mask_in]).sum().item()
            total += mask_in.sum().item()
            
        
        mask_out = ood_label == 1
        mask_out = mask_out.cpu()
        if mask_out.any():
            _out_score.extend(score[mask_out].tolist())

        if mask_wrong.any():
            _out_score.extend(score[mask_wrong].tolist())

        
        

    accuracy = correct / total if total > 0 else 0
    return _in_score, _out_score, accuracy