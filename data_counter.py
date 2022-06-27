import numpy as np

path = "ETF/0038/TestData"
etf_list = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']

count = {
    0.0: 0,
    1.0: 0,
    2.0: 0
}
for etf in etf_list:
    data = np.load(f"{path}/y_{etf}.npy")
    print(data.shape)
    unique, counts = np.unique(data, return_counts=True)
    for key, value in dict(zip(unique, counts)).items():
        count[key] += value

print(count)
percentage = {
    0.0: 0.0,
    1.0: 0.0,
    2.0: 0.0
}
for key, value in count.items():
    percentage[key] = value / (count[0.0] + count[1.0] + count[2.0])

print(percentage)
