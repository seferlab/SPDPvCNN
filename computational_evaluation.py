from architectures.helpers.constants import threshold
from architectures.helpers.model_handler import get_model
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf


MODEL_PATH = "1653380853-fdr0.25-sdr0.5-k5-e"
THRESHOLD = threshold
hyperparameters = hyperparameters[selected_model]
i = 80
run = "neat-planet-23"


def load_dataset():
    x_test = []
    y_test = []
    for etf in etf_list:
        x_test.extend(np.load(f"ETF/{threshold}/TestData/x_{etf}.npy"))
        y_test.extend(np.load(f"ETF/{threshold}/TestData/y_{etf}.npy"))
    return x_test, y_test


def make_datasets(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(hyperparameters["batch_size"])
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_finalized_datasets(x_test, y_test):
    test_dataset = make_datasets(x_test, y_test)
    return test_dataset


x_test, y_test = load_dataset()
test_dataset = get_finalized_datasets(x_test, y_test)
model = get_model()
model.load_weights(
    f"saved_models/{selected_model}/{THRESHOLD}/{run}/{MODEL_PATH}{i}.h5")
model.evaluate(test_dataset)
predictions = model.predict(test_dataset)
classes = np.argmax(predictions, axis=1)
cf = confusion_matrix(y_test, classes)
print(f"\n{cf}")
cr = classification_report(y_test, classes)
print(f"\n{cr}")
f1 = f1_score(y_test, classes, average='weighted')
print(f"\nF1 score: {f1}")
rc = recall_score(y_test, classes, average='weighted')
print(f"\nRecall score: {rc}")
pr = precision_score(y_test, classes, average='weighted')
print(f"\nPrecision score: {pr}")
