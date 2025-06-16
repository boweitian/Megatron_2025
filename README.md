# 🧠 Clean-label Backdoor Attacks on Vision Transformers

This repository implements clean-label backdoor attacks on Vision Transformers (ViTs), as introduced in:

> **Backdoor Attacks Meet Vision Transformers: Exploiting Attention for Clean-label Poisoning**  
> 📄 Paper: [arXiv:2412.04776](https://arxiv.org/abs/2412.04776)

## 📌 Overview

This project proposes a clean-label poisoning method for ViTs by jointly optimizing:

- **Latent loss**: aligning features of poisoned and target images.
- **Attention loss**: forcing attention to focus on the trigger.
- **PCGrad**: multi-objective gradient projection to reduce conflict.

The attack remains stealthy (high SSIM/LPIPS), yet effective (high ASR).

## 🧩 Code Structure

### `trainAndTest.py`
- `train()`: Supports both clean and poisoned training. Integrates latent loss, attention loss, and PCGrad.
- `valSet_poisonData_test()`: Evaluates attack success rate (ASR).
- `addMeasure_valSet_poisonData_test()`: Computes LPIPS, SSIM, PSNR for perceptual stealthiness.
- `BAVT_defense()`: Optional defense module.

## ⚙️ Environment Setup & Quick Start

Follow these simple steps to set up the environment and reproduce the results:

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/boweitian/Megatron_2025.git
cd Megatron_2025 
```

### ✅ Step 2: Create the Conda Environment
We provide a full environment specification in vit2025.yaml:

```bash
conda env create -f vit2025.yaml
conda activate vit2025
```

This will install all required packages including:

PyTorch 1.13.0 + CUDA 11.6

torchvision, timm

lpips, pytorch-msssim, opencv-python

tqdm, matplotlib, scikit-learn

✅ Step 3: Run the Demo Script

```bash
bash demonstrate_batch
```

This will launch a batch demo showcasing the poisoning attack and evaluation pipeline.

## 📌 Citation
If you find this project useful, please also consider citing our follow-up work:

```bibtex
@misc{gong2024megatronevasivecleanlabelbackdoor,
  title={Megatron: Evasive Clean-Label Backdoor Attacks against Vision Transformer},
  author={Xueluan Gong and Bowei Tian and Meng Xue and Shuike Li and Yanjiao Chen and Qian Wang},
  year={2024},
  eprint={2412.04776},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.04776},
}
```