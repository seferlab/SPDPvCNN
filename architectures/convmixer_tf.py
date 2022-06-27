import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
import architectures.helpers.constants as constants
hyperparameters = constants.hyperparameters["convmixer_tf"]

"""REFERENCE: (https://github.com/dmezh/convmixer-tf/blob/main/ConvMixer.py)"""


class Residual(keras.layers.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x


def GELU():
    return keras.layers.Activation(tf.keras.activations.gelu)


def ConvMixer(dim, depth, kernel_size, patch_size, n_classes):
    return keras.Sequential(
        [
            keras.layers.Conv2D(
                dim,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                input_shape=hyperparameters["input_shape"],
            ),
            GELU(),
            keras.layers.BatchNormalization(),
            *[
                keras.Sequential(
                    [
                        Residual(
                            keras.Sequential(
                                [
                                    keras.layers.Conv2D(
                                        dim,
                                        kernel_size=(kernel_size, kernel_size),
                                        groups=dim,
                                        padding="same",
                                    ),
                                    GELU(),
                                    keras.layers.BatchNormalization(),
                                ]
                            )
                        ),
                        keras.layers.Conv2D(dim, kernel_size=(1, 1)),
                        GELU(),
                        keras.layers.BatchNormalization(),
                    ]
                )
                for _ in range(depth)
            ],
            tfa.layers.AdaptiveAveragePooling2D((1, 1)),
            keras.layers.Flatten(),
            keras.layers.Activation(tf.keras.activations.linear),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


def compile_model_optimizer(model):
    optimizer = keras.optimizers.Adadelta()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_cm_tf_model():
    print("Getting the ConvMixer TF model...")
    conv_mixer_model = ConvMixer(
        hyperparameters["filters"], hyperparameters["depth"], hyperparameters["kernel_size"], hyperparameters["patch_size"], 3)
    return compile_model_optimizer(conv_mixer_model)
