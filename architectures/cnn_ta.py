from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import tensorflow_addons as tfa

import architectures.helpers.constants as constants

hyperparameters = constants.hyperparameters["cnn_ta"]


def get_ct_model():
    # optimizer = keras.optimizers.Adadelta()
    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=hyperparameters["learning_rate"], weight_decay=0.0001
    # )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.001)

    print("Getting the CNN_TA model...")
    model = Sequential()
    model.add(Conv2D(32, hyperparameters["kernel_size"], activation='relu',
                     input_shape=hyperparameters["input_shape"]))
    model.add(Conv2D(64, hyperparameters["kernel_size"], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hyperparameters["first_dropout_rate"]))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(hyperparameters["second_dropout_rate"]))
    model.add(Dense(hyperparameters["num_classes"], activation='softmax'))

    sam_model = SAMModel(model)

    sam_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=[
                          keras.metrics.SparseCategoricalAccuracy(
                              name="accuracy"),
                          keras.metrics.SparseTopKCategoricalAccuracy(
                              5, name="top5-acc"),
                      ],
                      )
    return sam_model


class SAMModel(tf.keras.Model):
    def __init__(self, inner_model, rho=0.02):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.inner_model = inner_model
        self.rho = rho

    def call(self, inputs):
        return self.inner_model(inputs)

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.inner_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.inner_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.inner_model(images)
            loss = self.compiled_loss(labels, predictions)

        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.inner_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm
