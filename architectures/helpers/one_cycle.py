"""
OneCycle Learning Rate Scheduler
Dan Mezhiborsky - @dmezh
See:
https://github.com/dmezh/convmixer-tf
https://github.com/tmp-iclr/convmixer/issues/11#issuecomment-951947395
"""

import numpy as np
from tensorflow import keras


class OneCycleLRScheduler(keras.callbacks.Callback):
    def __init__(self, epoch_count, lr_max, batches_per_epoch):
        super().__init__()
        self.epoch_count = epoch_count
        self.epoch = 1
        self.lr_max = lr_max
        self.batches_per_epoch = batches_per_epoch

    def on_batch_begin(self, batch: int, logs=None):
        self.batch = batch
        self.t = self.epoch + (self.batch + 1) / self.batches_per_epoch
        sched = np.interp(
            [self.t],
            [0, self.epoch_count * 2 // 5,
                self.epoch_count * 4 // 5, self.epoch_count],
            [0, self.lr_max, self.lr_max / 20.0, 0],
        )[0]
        keras.backend.set_value(self.model.optimizer.lr, sched)

    def on_epoch_begin(self, epoch: int, logs=None):
        epoch = epoch + 1  # tensorflow is off-by-one :P
        self.epoch = epoch
        print(
            f"lr at epoch {epoch}: {keras.backend.get_value(self.model.optimizer.lr)}"
        )
