# Graph Neural Network on PubMed with PyTorch Geometric

This repository provides a PyTorch Geometric implementation of a Graph Convolutional Network (GCN) on the PubMed citation dataset. The project includes dataset loading, model training, evaluation, and visualization of node embeddings using t-SNE.

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

- Loading and normalizing the PubMed dataset using PyTorch Geometric's `Planetoid` class.
- Defining a simple two-layer Graph Convolutional Network (GCN).
- Training the GCN model using node classification loss.
- Visualizing the learned node embeddings using t-SNE.
- Evaluating the model using classification accuracy.

## Requirements

- Python 3.8+
- torch
- torch-geometric
- scikit-learn
- seaborn
- matplotlib

Install the required packages with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn seaborn matplotlib
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gnn-pubmed-pyg.git
cd gnn-pubmed-pyg
```

Ensure dependencies are installed and your environment is properly activated.

## Dataset

The dataset used is **PubMed**, a citation network of scientific publications. Each publication is described by a sparse bag-of-words feature vector and classified into one of three classes.

PyTorch Geometric will download and prepare the dataset automatically at:

```
data/Planetoid/PubMed/
```

## Usage

To train the model and visualize embeddings, run:

```bash
python graph_intro.py
```

This script will:

- Load and normalize the dataset
- Define and train a GCN model
- Plot training loss over epochs
- Compute test accuracy
- Visualize node embeddings using t-SNE

## Script Details

### Data Preprocessing

- Dataset: `Planetoid(name='PubMed')`
- Feature Normalization: `NormalizeFeatures()`

### Model Architecture

```python
class GCN(torch.nn.Module):
    def __init__(self, num_hidden, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Training Details

- Optimizer: Adam (default lr)
- Loss: CrossEntropyLoss
- Epochs: 1000

### Key Outputs

- `loss_lst`: list of training losses
- `test_acc`: final test accuracy
- t-SNE scatterplot of test node embeddings colored by class

## Evaluation

After training, the model:

- Prints the final test set accuracy
- Visualizes node embeddings using `sklearn.manifold.TSNE`
- Displays a line plot of training loss using `seaborn`

## License

This project is licensed under the MIT License.