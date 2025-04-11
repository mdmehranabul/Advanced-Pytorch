# Convolutional Autoencoder with PyTorch

This repository provides a PyTorch implementation of a simple convolutional autoencoder trained on grayscale images. The model demonstrates how to compress images into a lower-dimensional latent space and reconstruct them back. It includes image preprocessing, model architecture, training loop, and visualizations of the original, latent, and reconstructed images.

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

This project implements:
A convolutional encoder that compresses input images into a 128-dimensional vector.
A convolutional decoder that reconstructs the original image from this vector.
A training loop that minimizes MSE loss between input and output.
Visualization of original, latent space representations, and reconstructed outputs.

## Requirements

Python 3.8+
torch
torchvision
numpy
matplotlib

Install the required packages with:
```bash
pip install torch torchvision numpy matplotlib
```

## Installation

Clone the repository:

git clone https://github.com/your-username/conv-autoencoder-pytorch.git
cd conv-autoencoder-pytorch

Make sure the required dependencies are installed.

## Dataset

Your dataset should be organized in the following folder structure:

data/train/
├── class1/
│   ├── image1.jpg
│   └── ...
├── class2/
│   ├── image2.jpg
│   └── ...

Each class subfolder can contain any number of images. Labels are not used in the training process.

## Usage

To train and visualize the autoencoder, run:

```bash
python Autoencoders.py
```
This will:

Load and preprocess the images

Train the autoencoder for 30 epochs

Show original images, latent vectors (reshaped), and reconstructed images

Print the achieved compression rate

## Script Details

Transformations

The following image transformations are applied:

Resize to 64x64

Convert to grayscale (1 channel)

Normalize to [-1, 1]

Model Architecture

Encoder:

Conv2d: 1 -> 6 channels (kernel size 3)

Conv2d: 6 -> 16 channels (kernel size 3)

ReLU

Flatten

Fully connected: 57600 -> 128 (latent space)

Decoder:

Fully connected: 128 -> 57600

Reshape to (16, 60, 60)

ConvTranspose2d: 16 -> 6 (kernel size 3)

ReLU

ConvTranspose2d: 6 -> 1 (kernel size 3)

ReLU

Training Loop

Optimizer: Adam (lr = 0.001)

Loss: Mean Squared Error (MSE)

Epochs: 30

## Evaluation

During and after training, the script:

Prints average loss per epoch

Visualizes:

Original input images

Latent vectors (reshaped to 8x16)

Reconstructed outputs

Compression Rate

Compression rate is computed as:

compression_rate = (1 - LATENT_DIM / (image_height * image_width)) * 100

With 64x64 input images and latent dimension 128, this results in about 96.88% compression.

## License

This project is licensed under the MIT License.