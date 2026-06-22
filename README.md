# 🔬 Monkeypox Skin Lesion Classification

> **Robust Multi-Class Monkeypox Classification Using a Calibrated Hybrid Deep Learning Ensemble**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/zainabatef)

---

## 📌 Overview

This repository contains the implementation of a hybrid deep learning ensemble for multi-class skin lesion classification, targeting monkeypox and visually similar diseases. The framework combines three pretrained convolutional backbone networks with temperature scaling calibration and Test-Time Augmentation (TTA).

**Classes:** Monkeypox · Chickenpox · Measles · Normal

---

## 🏗️ Model Architecture

| Component | Details |
|---|---|
| Backbone 1 | EfficientNetB3 (ImageNet pretrained) |
| Backbone 2 | DenseNet201 (ImageNet pretrained) |
| Backbone 3 | ConvNeXt-Small (ImageNet-22K → 1K) |
| Ensemble | Weighted logit averaging (0.8 / 0.2) |
| Calibration | Temperature Scaling (Adam-based NLL minimization) |
| Inference | TTA × 5 (original, hflip, vflip, hvflip, center-crop) |

---

## 📂 Repository Structure

```
monkeypox-classification/
│
├── monkeypox_classification.ipynb   # Main training & evaluation notebook
├── README.md                        # Project documentation
└── LICENSE                          # MIT License
```

---

## 📊 Dataset

- **Name:** MSID (Monkeypox Skin Images Dataset)
- **Total Images:** 770
- **Split:** 80% Train / 10% Validation / 10% Test (Stratified)

| Class | Images |
|---|---|
| Monkeypox | 279 |
| Chickenpox | 107 |
| Measles | 91 |
| Normal | 293 |

> Dataset available on Kaggle: [MSID Dataset](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msid)

---

## ⚙️ Key Techniques

- **Loss Function:** Focal Loss (γ = 1.5) with class weights
- **Optimizer:** AdamW with cosine annealing LR
- **Regularization:** EMA (decay = 0.999), label smoothing (α = 0.1)
- **Augmentation:** Affine, HueSaturation, GaussNoise, CoarseDropout, VerticalFlip
- **Pseudo-labeling:** Confidence threshold ≥ 0.70
- **Cross-validation:** 5-Fold Stratified K-Fold

---

## 🚀 How to Run

### Option 1: Kaggle (Recommended)
1. Open the notebook directly on Kaggle
2. Add the MSID dataset as input
3. Run all cells (GPU T4 x2 recommended)

### Option 2: Local
```bash
# Clone the repository
git clone https://github.com/zainabatef/monkeypox-classification.git
cd monkeypox-classification

# Install dependencies
pip install torch torchvision timm albumentations pandas numpy scikit-learn

# Launch notebook
jupyter notebook monkeypox_classification.ipynb
```

---

## 📦 Requirements

```
torch >= 2.0
torchvision >= 0.15
timm >= 0.9
albumentations >= 1.3
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
```

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{atef2024monkeypox,
  author       = {Zainab Atef and Mostafa R. Kaseb and Mohamed Hassan Farrag and Ahmed Abdelhafeez},
  title        = {Robust Multi-Class Monkeypox Classification Using a Calibrated Hybrid Deep Learning Ensemble},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/zainabatef/monkeypox-classification}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
