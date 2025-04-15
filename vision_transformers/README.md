# Dog Breed Classification with Vision Transformer (ViT)

This repository provides a PyTorch-based implementation for classifying dog breeds using a Vision Transformer (ViT) model, powered by HugsVision. It includes dataset loading, training, evaluation with a confusion matrix, and inference (both single image and batch).

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

This project demonstrates fine-tuning the pre-trained `google/vit-base-patch16-224-in21k` model from Hugging Face Transformers for image classification. It leverages the HugsVision library to streamline training and inference. A confusion matrix is generated to visualize performance, and the model is evaluated using the F1-score.

## Requirements

Python 3.8+  
torch  
transformers  
hugsvision  
scikit-learn  
matplotlib  
seaborn  
tqdm  
pandas  

Install dependencies via:

```bash
pip install torch transformers hugsvision scikit-learn matplotlib seaborn tqdm pandas
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/dog-breed-classifier.git
cd dog-breed-classifier
```

## Dataset

Organize your data in the following structure under a `train/` directory:

```
train/
├── affenpinscher/
│   ├── image1.jpg
│   └── image2.jpg
├── beagle/
│   ├── image1.jpg
│   └── image2.jpg
...
```

Optional test images for inference should be placed under:

```
test/
├── affenpinscher/
│   └── affenpinscher_0.jpg
...
```

## Usage

Run the training, evaluation, and inference pipeline:

```bash
python vit_classifier.py
```

This will:

1. Load and preprocess the image dataset.
2. Fine-tune a Vision Transformer model.
3. Evaluate using F1-score and confusion matrix.
4. Save predictions and visualize a confusion matrix.
5. Perform inference on individual and batch test images.

## Script Details

- **VisionDataset.fromImageFolder**: Loads dataset from folder structure and splits into train/test.
- **VisionClassifierTrainer**: Handles training, evaluation, and feature extraction.
- **ViTForImageClassification**: Pretrained ViT model adapted for dog breed classification.
- **VisionClassifierInference**: Wrapper for making predictions using trained model.
- **Confusion Matrix**: A heatmap is generated and saved as `conf_matrix_1.jpg`.

## Evaluation

- **F1-Score**: Evaluates model accuracy considering both precision and recall.
- **Confusion Matrix**: Highlights classification performance across all dog breeds.
- **Visual Output**: Saved as a heatmap image file.

## Inference

- **Single Image**:
  ```python
  predicted_label = classifier.predict(img_path="./test/affenpinscher/affenpinscher_0.jpg")
  ```

- **Batch Inference**:
  ```python
  for img_path in test_files:
      label = classifier.predict(img_path=img_path)
  ```

## License

This project is released under the MIT License.