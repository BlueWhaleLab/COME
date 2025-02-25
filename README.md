# COME: Test-time Adaption by Conservatively Minimizing Entropy (ICLR'25)

This is the official implementation of [COME: Test-time Adaption by Conservatively Minimizing Entropy](https://arxiv.org/abs/2410.10894) on ICLR 2025. We propose Conservatively Minimizating Entropy (COME) as a simple drop-in refinement of Entropy Minimization for test-time adaption.

## Installation Requirements

To get started with this repository, you need to follow these installation.

```
pip install -r requirements.txt
```

## Data preparation

We follow [Robustness bench](https://github.com/hendrycks/robustness) and [OpenOOD](https://github.com/Jingkang50/OpenOOD) to prepare the datasets. We provide the links to download each dataset:

- ImageNet-C: download it from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc).

The following datasets are only used in open-world TTA setting:

- iNaturalist: download it from [this link](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz)
- NINCO, SSB_Hard, Texture, Open-ImageNet: download them from [this link](https://drive.google.com/drive/folders/1IFb4pPWTHsvWV6ezzbmGkIR64_VnOdSh?usp=drive_link)

## Usage Example

COME can be implemtented by simply replacing the loss function of previous TTA algorithms i.e., Tent, EATA, and SAR from **softmax entropy** to **entropy of opinion**.

```python
def entropy_of_opinion(x: torch.Tensor): #key component of COME
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = K / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy

 def forward_and_adapt(x, model, optimizer, args):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    # COME: replace softmax_entropy with entropy_of_opinion
    loss = entropy_of_opinion(outputs) 
    loss = loss.mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs
```

## Reproduce the results

### Baselines

1. **no_adapt (source)**: The original model without any adaptation.
2. **Tent** & **EATA** & **SAR**: Previous TTA methods using Entropy Minimization.
3. **Tent_COME** & **EATA_COME** & **SAR_COME**: The enhanced COME version.

### Running the Code

To run the experiments, execute the script [start.sh](./start.sh).

### Results

Classification Accuracy Comparison on ImageNet-C (Level 5). Substantial (≥ 0.5) improvements compared to the baseline are marked with +.

| Methods  | COME    | Gauss. | Shot | Impul. | Defoc | Glass | Motion | Zoom | Snow  | Frost | Fog  | Brit. | Contr. | Elast. | Pixel | JPEG | Avg. Acc↑ |
| -------- | ------- | ------ | ---- | ------ | ----- | ----- | ------ | ---- | ----- | ----- | ---- | ----- | ------ | ------ | ----- | ---- | --------- |
| no_adapt | ✗       | 35.1   | 32.2 | 35.9   | 31.4  | 25.3  | 39.4   | 31.6 | 24.5  | 30.1  | 54.7 | 64.5  | 49.0   | 34.2   | 53.2  | 56.5 | 39.8      |
| **Tent** | ✗       | 52.4   | 51.8 | 53.3   | 53.0  | 47.6  | 56.8   | 47.6 | 10.6  | 28.0  | 67.5 | 74.2  | 67.4   | 50.2   | 66.7  | 64.6 | 52.8      |
|          | ✓       | 53.8   | 53.7 | 55.3   | 55.7  | 51.7  | 59.7   | 52.7 | 59.0  | 61.7  | 71.3 | 78.2  | 68.7   | 57.7   | 70.5  | 68.2 | 61.2      |
|          | Improve | +1.4   | +1.9 | +1.9   | +2.7  | +4.1  | +2.9   | +5.0 | +48.4 | +33.6 | +3.9 | +4.0  | +1.3   | +7.5   | +3.8  | +3.6 | +8.4      |
| **EATA** | ✗       | 55.9   | 56.5 | 57.1   | 54.1  | 53.3  | 61.9   | 58.7 | 62.1  | 60.2  | 71.3 | 75.4  | 68.5   | 62.8   | 69.3  | 66.6 | 62.2      |
|          | ✓       | 56.2   | 56.6 | 57.2   | 58.1  | 57.6  | 62.5   | 59.5 | 65.5  | 63.9  | 72.5 | 78.1  | 69.7   | 66.5   | 72.4  | 70.7 | 64.5      |
|          | Improve | +0.3   | +0.2 | +0.1   | +4.1  | +4.3  | +0.6   | +0.7 | +3.5  | +3.7  | +1.2 | +2.7  | +1.2   | +3.7   | +3.1  | +4.0 | +2.2      |
| **SAR**  | ✗       | 52.7   | 52.1 | 53.6   | 53.5  | 48.9  | 56.7   | 48.8 | 22.5  | 51.9  | 67.5 | 73.4  | 66.8   | 52.7   | 66.3  | 64.5 | 55.5      |
|          | ✓       | 56.2   | 56.5 | 57.5   | 58.3  | 56.7  | 62.9   | 58.2 | 65.3  | 64.8  | 72.6 | 78.5  | 69.3   | 64.4   | 71.9  | 69.5 | 64.2      |
|          | Improve | +3.5   | +4.4 | +3.8   | +4.8  | +7.7  | +6.2   | +9.5 | +42.9 | +12.8 | +5.0 | +5.1  | +2.5   | +11.6  | +5.6  | +5.0 | +8.7      |



## Citation

If you find COME helpful in your research, please consider citing our paper:

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

This repo is developed upon [SAR 🔗](https://github.com/mr-eggplant/SAR).



For any additional questions, feel free to email [qingyangzhang@tju.edu.cn](mailto:qingyangzhang@tju.edu.cn).