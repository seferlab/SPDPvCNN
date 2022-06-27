from architectures.helpers.constants import threshold
from architectures.helpers.model_handler import get_model
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model

import numpy as np
import tensorflow as tf


MODEL_PATH = "1655761185-tl8-pd64-p8-e"
THRESHOLD = threshold
hyperparameters = hyperparameters[selected_model]


class Wallet:
    def __init__(self, base_currency_name: str, stock_name: str, initial_money: float):
        self.base_currency_name: str = base_currency_name
        self.stock_name: str = stock_name
        self.initial_money: float = initial_money
        self.info: dict = {base_currency_name: initial_money, stock_name: 0, f"v_{base_currency_name}": initial_money, f"v_{stock_name}": 0,
                           "buy_count": 0, "hold_count": 0, "sell_count": 0}
        self.profit_percentage: float = 0
        #self.transactions: list = []

    def buy(self, stock_price: float, date: str):
        if self.info[self.base_currency_name] == 0:
            return
        self.info["buy_count"] += 1
        v_base = (self.info[self.base_currency_name] - 1)
        stock = v_base / stock_price
        # print(
        #     f"Bought {self.stock_name}: {round(stock, 2)} | USD: 0 | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.stock_name] = stock
        self.info[f"v_{self.stock_name}"] = stock
        self.info[self.base_currency_name] = 0
        self.info[f"v_{self.base_currency_name}"] = v_base
        self.profit_percentage = v_base / self.initial_money - 1

    def hold(self, stock_price: float):
        self.info["hold_count"] += 1
        self.update_values(stock_price)
        return

    def sell(self, stock_price: float, date: str):
        if self.info[self.stock_name] == 0:
            return
        self.info["sell_count"] += 1
        base = self.info[self.stock_name] * stock_price - 1
        v_stock = base / stock_price
        # print(
        #     f"Sold   {self.stock_name}: 0 | USD: {round(base, 2)} | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.base_currency_name] = base
        self.info[f"v_{self.base_currency_name}"] = base
        self.info[self.stock_name] = 0
        self.info[f"v_{self.stock_name}"] = v_stock
        self.profit_percentage = base / self.initial_money - 1

    def print_values(self):
        # if(self.profit_percentage > 0):
        print(self.info)
        print(f"Profit percentage: {self.profit_percentage/4}")

    def update_values(self, stock_price: float):
        if self.info[self.stock_name] > 0:
            self.info[f"v_{self.base_currency_name}"] = self.info[self.stock_name] * stock_price
        elif self.info[self.base_currency_name] > 0:
            self.info[f"v_{self.stock_name}"] = self.info[self.base_currency_name] / stock_price
        else:
            print("Error")
        self.profit_percentage = self.info[f"v_{self.base_currency_name}"] / \
            self.initial_money - 1


def load_dataset():
    x_test = []
    y_test = []
    for etf in etf_list:
        x_test.append(np.load(f"ETF/{THRESHOLD}/TestData/x_{etf}.npy"))
        y_test.append(np.load(f"ETF/{THRESHOLD}/TestData/y_{etf}.npy"))
    return x_test, y_test


def make_dataset(x_test, y_test):
    datasets = []
    # keeps the images and labels for every stock one by one (datasets[0] == images & labels for etf_list[0])
    for xt, yt in zip(x_test, y_test):
        dataset = tf.data.Dataset.from_tensor_slices((xt, yt))
        dataset = dataset.batch(hyperparameters["batch_size"])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        datasets.append(dataset)
    return datasets


"""Loading the necessary stuff"""

listOfDates: list[np.ndarray] = []
listOfPrices: list[np.ndarray] = []
# keeps the prices for every stock one by one (listOfPrices[0] == prices for etf_list[0])

for etf in etf_list:
    listOfDates.append(
        np.load(f"ETF/{THRESHOLD}/Date/TestDate/{etf}.npy", allow_pickle=True))
    listOfPrices.append(
        np.load(f"ETF/{THRESHOLD}/Price/TestPrice/{etf}.npy", allow_pickle=True))


x_test, y_test = load_dataset()
datasets = make_dataset(x_test, y_test)

profit_ranking = []

for i in [99]:
    model = get_model()
    model.load_weights(
        f"saved_models/{selected_model}/{THRESHOLD}/drawn-sky-108/{MODEL_PATH}{i}.h5")
    listOfSignals = []
    for dataset in datasets:
        predictions = model.predict(dataset)
        listOfSignals.append(np.argmax(predictions, axis=1))

    print(f"MODEL{i}")
    """Main algorithm"""
    profits = []
    daily_moneys = []
    for signals, etf, price, dates in zip(listOfSignals, etf_list, listOfPrices, listOfDates):
        wallet = Wallet("USD", etf, 10000)
        daily_money = []
        for signal, price, date in zip(signals, price, dates):
            if signal == 0:
                wallet.buy(price, date)
            elif signal == 1:
                wallet.hold(price)
            elif signal == 2:
                wallet.sell(price, date)
            daily_money.append(wallet.info[f"v_{wallet.base_currency_name}"])
        wallet.print_values()
        # print("\n")
        profits.append(wallet.profit_percentage/4)
        daily_moneys.append(daily_money)
    mpp = np.mean(profits)
    std_ = np.std(profits)
    # calculate log return of the profits
    log_returns = np.log(1 + np.array(profits))
    # calculate the sharpe ratio
    sharpe_ratio = np.mean(log_returns) / np.std(log_returns)

    print(f"Model mean profit percentage: {mpp}")
    print(f"Model sharpe ratio: {sharpe_ratio}")
    print(f"std: {std_}\n")
    profit_ranking.append(
        {"mpp": mpp, "model": i, "sharpe_ratio": sharpe_ratio})
    # for pr in profits:
    #     print(pr)

sorted_pr = sorted(
    profit_ranking, key=lambda d: d['sharpe_ratio'], reverse=True)
print(sorted_pr)
"""create a list of model values from sorted_pr"""
model_values = [d['model'] for d in sorted_pr]
print(model_values)

sorted_pr_ = sorted(
    profit_ranking, key=lambda d: d['mpp'], reverse=True)
print(sorted_pr_)
"""create a list of model values from sorted_pr"""
model_values_ = [d['model'] for d in sorted_pr_]
print(model_values_)

# for dm in daily_moneys:
#     print(dm)
