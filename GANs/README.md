# Heart Curve GAN with PyTorch

This repository provides a PyTorch implementation of a simple Generative Adversarial Network (GAN) trained to replicate a 2D heart-shaped curve. The project illustrates how a GAN can learn to mimic a specific data distribution, in this case, an artistic heart shape derived from parametric equations.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Script Details](#script-details)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

This project demonstrates:

A generator network that learns to produce 2D points resembling a heart-shaped curve.

A discriminator network that learns to distinguish between real points from the curve and fake points from the generator.

Alternating training of the discriminator and generator to reach an adversarial equilibrium.

Visualization of the generator's progress in mimicking the heart curve over epochs.

## Requirements

Python 3.8+
pip install torch numpy matplotlib seaborn
Installation

Clone the repository:
```bash
git clone https://github.com/your-username/heart-curve-gan.git
cd heart-curve-gan
```
## Dataset

The dataset is synthetically generated from a parametric equation of a heart shape:

x = 16 * (sin(theta))**3
y = 13*cos(theta) - 5*cos(2*theta) - 2*cos(3*theta) - cos(4*theta)

A total of 1024 points are sampled from theta in the range [0, 2*pi].

## Usage

To train the GAN and visualize progress, run:

python GAN.py

This will:

Generate 2D heart-shaped training data

Train the GAN for 8000 epochs

Save visualizations of generator output every 11 epochs in the train_progress/ directory

## Script Details

Generator

Input: 2D latent vector sampled from Gaussian distribution
Architecture: Linear(2->16) -> ReLU -> Linear(16->64) -> ReLU -> Linear(64->2)

Discriminator

Input: 2D sample (real or fake)

Architecture:

Linear(2->256) -> ReLU -> Dropout

Linear(256->128) -> ReLU -> Dropout

Linear(128->64) -> ReLU -> Dropout

Linear(64->1) -> Sigmoid

Training
Optimizers: Adam (default LR=0.001)
Loss: Binary Cross Entropy
Training alternates between:

Discriminator update (even epochs)

Generator update (odd epochs)

Visualization

Every 11 epochs, a scatter plot of 1000 generated points is saved
Outputs are saved to train_progress/imageXXX.jpg

## Evaluation

Progress is visualized qualitatively by plotting generated 2D samples. Over time, the generator learns to align its output distribution with the true heart curve.
Quantitative evaluation isn't applied in this simple GAN, but visual convergence gives an intuitive sense of learning progress.

## License

This project is licensed under the MIT License.
Feel free to open issues or PRs to suggest improvements or extensions!

