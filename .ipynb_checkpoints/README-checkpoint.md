# COME: Test-time Adaption by Conservatively Minimizing Entropy

This is the project repository for [COME: Test-time Adaption by Conservatively Minimizing Entropy](https://arxiv.org/abs/2410.10894) by Qingyang Zhang, Yatao Bian, Xinke Kong, Peilin Zhao, Changqing Zhang(ICLR 2025).

Machine learning models must continuously self-adjust themselves for novel data distribution in the open world. As the predominant principle, entropy minimization (EM) has been proven to be a simple yet effective cornerstone in existing test-time adaption (TTA) methods. While unfortunately its fatal limitation (i.e., overconfidence) tends to result in model collapse. For this issue, we propose to conservatively minimize the entropy (COME), which is a simple drop-in replacement of traditional EM to elegantly address the limitation. In essence, COME explicitly models the uncertainty by characterizing a Dirichlet prior distribution over model predictions during TTA. By doing so, COME naturally regularizes the model to favor conservative confidence on unreliable samples. Theoretically, we provide a preliminary analysis to reveal the ability of COME in enhancing the optimization stability by introducing a data-adaptive lower bound on the entropy. Empirically, our method achieves state-of-the-art performance on commonly used benchmarks, showing significant improvements in terms of classification accuracy and uncertainty estimation under various settings including standard, life-long and open-world TTA.

We provide **[example code](#1)** in PyTorch to illustrate the **COME** method and fully test-time adaptation setting.

We provide implementations of the classic EM algorithms—Tent, EATA, and SAR—along with the enhanced COME version. Additionally, we offer two scripts designed for running both fully test-time adaptation setting and open-world test-time adaptation settings. You're welcome to experiment with your own datasets and models as well!
**Installation**:

```
pip install -r requirements.txt
```

**Data preparation**:

This repository contains code for evaluation on [ImageNet-C 🔗](https://arxiv.org/abs/1903.12261) with VitBase and ResNet. 

- Step 1: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

- Step 2: Put IamgeNet-C path at "--data_corruption" in main.py or "data_corruption" in start.sh.

- Step 3: Put output path at "--output" in main.py or "output" in start.sh.

- Step 4 [optional, for EATA]: Put ImageNet **test/val set**  at "--data" in main.py or "data" in start.sh.

- Step 5 [optional, for open-world setting]: Put NINCO, iNaturalist, SSB_Hard, Texture, Openimage_O at "--ood_root" in main.py or "ood_root" in start-open.sh.


**COME** depends on

- Python 3
- [PyTorch](https://pytorch.org/) >= 1.0


**Usage**:

```
import tent_come

net = backbone_net()
net = tent_come.configure_model(net)
params, param_names = tent_come.collect_params(net)
optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum) 
adapt_model = tent_come.Tent_COME(net, optimizer)

outputs = adapt_model(inputs)  # now it infers and adapts!
```
<span id="1">  <span>
## Example: TTA setting

This example demonstrates how to adapt an ImageNet1K classifier to handle image corruptions on the ImageNet_C dataset.

### Methods Compared

1. **no_adapt (source)**: The original model without any adaptation.
2. **Tent** & **EATA** & **SAR**: Classic EM algorithms adapt the model at test time using entropy minimization.
3. **Tent_COME** & **EATA_COME** & **SAR_COME**: Classic EM algorithms with the enhanced COME version adapt the model at test time using entropy minimization.

### Dataset

The dataset used is [ImageNet_C](https://github.com/hendrycks/robustness/), containing 15 corruption types, each with 5 levels of severity.

### Running the Code

To run the experiments, execute the script [start.sh](./start.sh).

### Result: Classification Accuracy Comparison on ImageNet-C (Level 5)

Substantial (≥ 0.5) improvements compared to the baseline are marked with +. We only report average FPR↓ in the appendix.

| Methods  | COME | Gauss. | Shot  | Impul. | Defoc | Glass | Motion | Zoom  | Snow  | Frost | Fog   | Brit. | Contr. | Elast. | Pixel | JPEG  | Avg. Acc↑ |
|----------|------|--------|-------|--------|-------|-------|--------|-------|-------|-------|-------|-------|--------|--------|-------|-------|-----------|
| no_adapt | ✗    | 35.1   | 32.2  | 35.9   | 31.4  | 25.3  | 39.4   | 31.6  | 24.5  | 30.1  | 54.7  | 64.5  | 49.0   | 34.2   | 53.2  | 56.5  | 39.8      |
| **Tent** | ✗    | 52.4   | 51.8  | 53.3   | 53.0  | 47.6  | 56.8   | 47.6  | 10.6  | 28.0  | 67.5  | 74.2  | 67.4   | 50.2   | 66.7  | 64.6  | 52.8      |
|          | ✓    | 53.8   | 53.7  | 55.3   | 55.7  | 51.7  | 59.7   | 52.7  | 59.0  | 61.7  | 71.3  | 78.2  | 68.7   | 57.7   | 70.5  | 68.2  | 61.2      |
|          |Improve| +1.4   | +1.9  | +1.9   | +2.7  | +4.1  | +2.9   | +5.0  | +48.4 | +33.6 | +3.9  | +4.0  | +1.3   | +7.5   | +3.8  | +3.6  | +8.4      |
| **EATA** | ✗    | 55.9   | 56.5  | 57.1   | 54.1  | 53.3  | 61.9   | 58.7  | 62.1  | 60.2  | 71.3  | 75.4  | 68.5   | 62.8   | 69.3  | 66.6  | 62.2      |
|          | ✓    | 56.2   | 56.6  | 57.2   | 58.1  | 57.6  | 62.5   | 59.5  | 65.5  | 63.9  | 72.5  | 78.1  | 69.7   | 66.5   | 72.4  | 70.7  | 64.5      |
|          |Improve| +0.3   | +0.2  | +0.1   | +4.1  | +4.3  | +0.6   | +0.7  | +3.5  | +3.7  | +1.2  | +2.7  | +1.2   | +3.7   | +3.1  | +4.0  | +2.2      |
| **SAR**  | ✗    | 52.7   | 52.1  | 53.6   | 53.5  | 48.9  | 56.7   | 48.8  | 22.5  | 51.9  | 67.5  | 73.4  | 66.8   | 52.7   | 66.3  | 64.5  | 55.5      |
|          | ✓    | 56.2   | 56.5  | 57.5   | 58.3  | 56.7  | 62.9   | 58.2  | 65.3  | 64.8  | 72.6  | 78.5  | 69.3   | 64.4   | 71.9  | 69.5  | 64.2      |
|          |Improve| +3.5   | +4.4  | +3.8   | +4.8  | +7.7  | +6.2   | +9.5  | +42.9 | +12.8 | +5.0  | +5.1  | +2.5   | +11.6  | +5.6  | +5.0  | +8.7      |

## Citation
If our COME method is helpful in your research, please consider citing our paper:
```
@inproceedings{zhang2025come,
  title={COME: TEST-TIME ADAPTION BY CONSERVATIVELY
MINIMIZING ENTROPY},
  author={Qingyang Zhang, Yatao Bian, Xinke Kong, Peilin Zhao, Changqing Zhang},
  booktitle = {Internetional Conference on Learning Representations},
  year = {2025}
}
```
## Acknowledgment
The code is inspired by the [SAR 🔗](https://github.com/mr-eggplant/SAR).

