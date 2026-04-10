# MLOps Lab 6 Environment

This project contains a clean environment for running an XGBoost + Weights & Biases workflow.

## Setup

Create and activate the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mlopslab6 --display-name "Python (mlopslab6)"
```

## Run Script

Run a quick local test without W&B:

```bash
python train_wine_xgboost.py
```

For online W&B logging from your normal terminal:

```bash
wandb login
python train_wine_xgboost.py --wandb-mode online
```

For local W&B files without syncing:

```bash
python train_wine_xgboost.py --wandb-mode offline
```

## Assignment :
Use the Wine dataset to answer:

> Can an XGBoost multiclass classifier identify wine cultivars from chemical measurements, and which hyperparameters improve test accuracy?

Suggested tasks:

1. Load the Wine dataset and describe its features/classes.
2. Split the data into train/test sets using stratification.
3. Train an XGBoost multiclass classifier.
4. Log hyperparameters, train/test metrics, and a confusion matrix to W&B.
5. Change at least two hyperparameters, such as `max_depth`, `eta`, or `rounds`.
6. Compare the runs and write a short conclusion about which settings worked best.


