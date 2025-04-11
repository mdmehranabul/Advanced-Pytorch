# Movie Recommendation System

This repository contains a PyTorch-based implementation of a simple matrix factorization recommendation system trained on user-movie ratings. It demonstrates data loading, model definition, training, evaluation, and basic ranking metrics (precision@k and recall@k).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Script Details](#script-details)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Overview 

The script loads a ratings.csv file containing user-movie ratings, encodes user and movie IDs, defines a MovieDataset, and trains a simple embedding-based model (RecSysModel) to predict ratings. After training, it computes Mean Squared Error (MSE) on a held-out test set and evaluates recommendation quality using precision@k and recall@k.

## Requirements

Python 3.8+
pandas
scikit-learn
torch (PyTorch) 2.2.0 or higher

Install dependencies via:

```bash
pip install pandas scikit-learn torch torchvision torchaudio
```
## Installation

Clone the repository:



git clone 
cd 

2. Ensure dependencies are installed (see [Requirements](#requirements)).

## Dataset
Place your `ratings.csv` file in the repository root. The expected format:

| userId | movieId | rating | timestamp |
| ------ | ------- | ------ | --------- |
| 1      | 1       | 4.0    | 964982703 |
| 1      | 3       | 4.0    | 964981247 |

- **userId**: integer user identifier
- **movieId**: integer movie identifier
- **rating**: float rating value
- **timestamp**: (unused) UNIX timestamp of rating

## Usage
Run the main training and evaluation script:

```bash
python MatrixFactorization_start.py
```
This will:

Load and preprocess data.

Split into training and test sets (70/30).

Train the RecSysModel for 1 epoch.

Compute and print MSE on the test set.

Compute and print precision@10 and recall@10.

## Script Details

MovieDataset: A torch.utils.data.Dataset that returns (user_tensor, movie_tensor, rating_tensor).

RecSysModel: A PyTorch nn.Module with two embedding layers (users and movies) and a linear output layer.

Training Loop: Uses nn.MSELoss and torch.optim.Adam to optimize embeddings.

Evaluation:

MSE: Mean Squared Error between predicted and true ratings.

Precision@k / Recall@k: For each user, recommends top-k movies by predicted rating, counts how many are truly relevant (rating >= 3.5).

## Evaluation

- Mean Squared Error: Measures average squared difference between predictions and true ratings.

- Precision@10: Fraction of top-10 recommendations that are truly relevant.

- Recall@10: Fraction of all relevant movies that appear in the top-10 recommendations.

## License

This project is released under the MIT License.