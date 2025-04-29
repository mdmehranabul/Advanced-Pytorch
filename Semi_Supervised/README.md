# Semi-Supervised Image Classification with PyTorch

This repository demonstrates a semi-supervised learning framework implemented using PyTorch. The model is trained on both labeled and unlabeled grayscale animal images (e.g., bears, pandas), learning not just class distinctions but also auxiliary self-supervised rotation tasks.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Script Details](#script-details)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

This project showcases how to:

- Load labeled and unlabeled image datasets
- Apply grayscale transformations and self-supervised rotation augmentation
- Train a neural network on both classification and auxiliary rotation prediction tasks
- Improve performance on limited labeled data using unlabeled data in a semi-supervised setting

The supervised task is a binary classification between `Bears` and `Pandas`. The self-supervised task is to predict the rotation angle (0°, 90°, 180°, 270°) of unlabeled images.

## Requirements

Python 3.8+  
torch  
torchvision  
Pillow  
scikit-learn  
seaborn  

Install dependencies via:

```bash
pip install torch torchvision pillow scikit-learn seaborn
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/semi-supervised-pytorch.git
cd semi-supervised-pytorch
```

## Dataset Structure

Your data directory should be structured as follows:

```
data/
├── train/
│   ├── Bears/
│   └── Pandas/
├── test/
│   ├── Bears/
│   └── Pandas/
└── unlabeled/
    ├── bear1.jpg
    ├── panda2.jpg
    └── ...
```

- `train/` and `test/` contain labeled images sorted by class.
- `unlabeled/` contains unlabeled images used for self-supervised training.

## Usage

To run the training pipeline:

```bash
python semi_supervised.py
```

This script will:

1. Load both supervised and self-supervised datasets
2. Train a convolutional neural network (`SesemiNet`) with two output heads:
   - One for binary classification (supervised)
   - One for rotation angle prediction (self-supervised)
3. Plot training loss across epochs
4. Evaluate classification accuracy on the test set

## Script Details

- **UnlabeledDataset**: Custom `Dataset` class that applies random 0°, 90°, 180°, or 270° rotations for self-supervision.
- **SesemiNet**: CNN with a shared backbone and two output heads (supervised + self-supervised).
- **Loss Composition**: Combines supervised classification loss and self-supervised rotation loss.

## Evaluation

At the end of training, the model is evaluated on the test set using classification accuracy. A line plot of the training loss is also generated using seaborn.

## License

This project is released under the MIT License.