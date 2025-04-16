# Linear Regression with PyTorch Lightning

This repository demonstrates a simple linear regression model implemented using PyTorch Lightning. It predicts car mileage (mpg) based on car weight (wt) using a dataset sourced from a public CSV.

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

This project showcases how to:

- Load a dataset using Pandas
- Convert it into PyTorch tensors
- Create a custom `Dataset` and `DataLoader`
- Build and train a linear regression model using PyTorch Lightning
- Track training progress using early stopping

The project uses the classic `cars` dataset where `wt` is used as the input feature and `mpg` as the output label.

## Requirements

Python 3.8+  
torch  
pandas  
numpy  
matplotlib  
seaborn  
pytorch-lightning  

Install dependencies via:

```bash
pip install torch pandas numpy matplotlib seaborn pytorch-lightning
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/linear-regression-pytorch-lightning.git
cd linear-regression-pytorch-lightning
```

## Dataset

The `cars` dataset is fetched directly from a public GitHub Gist:

```python
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/.../cars.csv'
cars = pd.read_csv(cars_file)
```

Features and labels are extracted and converted to PyTorch tensors for training.

## Usage

Run the training pipeline with:

```bash
python lit_linear_regression.py
```

This will:

1. Load and preprocess the dataset
2. Initialize a LightningModule for linear regression
3. Train the model with early stopping on training loss
4. Print the learned parameters

## Script Details

- **LinearRegressionDataset**: A PyTorch `Dataset` for pairing features with targets.
- **LitLinearRegression**: A PyTorch Lightning `LightningModule` for linear regression.
- **EarlyStopping Callback**: Stops training if training loss doesn't improve for 2 epochs.
- **Trainer**: Configured to use GPU acceleration if available and trains for a max of 1000 epochs.

## Evaluation

After training, the model's learned parameters (weights and bias) are printed. These can be manually inspected or used for further predictions.

## License

This project is released under the MIT License.