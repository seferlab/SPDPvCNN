import tensorflow_addons as tfa
import architectures.helpers.constants as constants
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow import keras
from keras import metrics

'''
CONVMIXER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/convmixer.py)
'''
hyperparameters = constants.hyperparameters["convmixer"]


''' ConvMixer Implementation
'''


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size,
                      strides=patch_size)(x)
    # , kernel_regularizer=regularizers.l2(1e-2)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    # ,
    #                   kernel_regularizer=regularizers.l2(1e-2)
    x = activation_block(x)

    return x


def get_conv_mixer_model(
        image_size=hyperparameters["image_size"], filters=hyperparameters["filters"], depth=hyperparameters["depth"],
        kernel_size=hyperparameters["kernel_size"], patch_size=hyperparameters["patch_size"], num_classes=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


''' Compiling, Training and Evaluating
'''


def compile_model_optimizer(model):
    # optimizer = keras.optimizers.Adadelta()
    optimizer = tfa.optimizers.AdamW(
        learning_rate=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"]
    )

    sam_model = SAMModel(model)

    sam_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"), ]
    )
    return sam_model

# Used to load a saved model to train and/or evaluate


def load_saved_model(path):
    return keras.models.load_model(path, custom_objects={'MyOptimizer': tfa.optimizers.AdamW})


def get_cm_model():
    print("Getting the ConvMixer model...")
    conv_mixer_model = get_conv_mixer_model()
    return compile_model_optimizer(conv_mixer_model)


class SAMModel(tf.keras.Model):
    def __init__(self, inner_model, rho=0.1):
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
