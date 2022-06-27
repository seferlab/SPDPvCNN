import numpy as np
import pandas as pd

from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model
from architectures.helpers.wandb_handler import initialize_wandb
from architectures.helpers.custom_callbacks import CustomCallback


def load_dataset():
    x_train = []
    y_train = []
    for etf in etf_list:
        x_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/x_{etf}.npy"))
        y_train.extend(
            np.load(f"ETF/strategy/{threshold}/TrainData/y_{etf}.npy"))
    return x_train, y_train


x_train, y_train = load_dataset()
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)

x_train_new = []
y_train_new = []

for x_t, y_t in zip(x_train, y_train):
    if y_t != 1:
        x_train_new.append(x_t)
        y_train_new.append(y_t)
        x_train_new.append(x_t)
        y_train_new.append(y_t)

x_train.extend(x_train_new)
y_train.extend(y_train_new)

unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)
