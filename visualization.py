import matplotlib.pyplot as plt
import numpy as np

from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold


def load_dataset():
    x_train = np.load(f"ETF/strategy/{threshold}/TrainData/x_SPY.npy")
    y_train = np.load(f"ETF/strategy/{threshold}/TrainData/y_SPY.npy")
    dates = np.load(
        f"ETF/strategy/{threshold}/Date/TrainDate/SPY.npy", allow_pickle=True)
    return x_train, y_train, dates


x_train, y_train, dates = load_dataset()
l = []

while len(l) < 3:
    ri = np.random.randint(0, 4010)
    # check if l contains y_train[ri]
    if y_train[ri] not in l:
        l.append(y_train[ri])
        print(f"{ri}: {y_train[ri]} at {dates[ri]}")
        plt.imshow(x_train[ri], cmap='gray')
        plt.show()
