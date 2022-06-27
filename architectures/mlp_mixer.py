import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import architectures.helpers.constants as constants

from tensorflow import keras
from tensorflow.keras import layers

hyperparameters = constants.hyperparameters["mlp_mixer"]

'''
IMPLEMENTING THE VISION TRANSFORMER
Reference: (https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py)
'''

# print(
#     f"Image size: {hyperparameters['image_size']} X {hyperparameters['image_size']} = {hyperparameters['image_size'] ** 2}")
# print(
#     f"Patch size: {hyperparameters['patch_size']} X {hyperparameters['patch_size']} = {hyperparameters['patch_size'] ** 2} ")
# print(f"Patches per image: {hyperparameters['num_patches']}")
# print(
#     f"Elements per patch (3 channels): {(hyperparameters['patch_size'] ** 2) * 3}")

"""
Build a classification model
We implement a method that builds a classifier given the processing blocks.
"""


def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=hyperparameters['input_shape'])
    # Create patches.
    patches = Patches(
        hyperparameters["patch_size"], hyperparameters["num_patches"])(inputs)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=hyperparameters["embedding_dim"])(patches)
    if positional_encoding:
        positions = tf.range(
            start=0, limit=hyperparameters["num_patches"], delta=1)
        position_embedding = layers.Embedding(
            input_dim=hyperparameters["num_patches"], output_dim=hyperparameters["embedding_dim"]
        )(positions)
        x = x + position_embedding
    # Process x using the module blocks.
    x = blocks(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(
        rate=hyperparameters["dropout_rate"])(representation)
    # Compute logits outputs.
    logits = layers.Dense(hyperparameters["num_classes"])(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)


"""
Define an experiment
We implement a utility function to compile, train, and evaluate a given model.
"""


def compile_model(model):
    # Create Adam optimizer with weight decay.
    optimizer = keras.optimizers.Adadelta()
    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    return model
    # Fit the model.
    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_split=0.1,
    #     callbacks=[early_stopping, reduce_lr],
    # )

    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # # Return history to plot learning curves.
    # return history


"""
Implement patch extraction as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(
            patches, [batch_size, self.num_patches, patch_dims])
        return patches


"""
## The MLP-Mixer model
The MLP-Mixer is an architecture based exclusively on
multi-layer perceptrons (MLPs), that contains two types of MLP layers:
1. One applied independently to image patches, which mixes the per-location features.
2. The other applied across patches (along channels), which mixes spatial information.
This is similar to a [depthwise separable convolution based model](https://arxiv.org/pdf/1610.02357.pdf)
such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization
instead of batch normalization.
"""

"""
Implement the MLP-Mixer module
"""


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=hyperparameters["embedding_dim"]),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


"""
The MLP-Mixer model tends to have much less number of parameters compared
to convolutional and transformer-based models, which leads to less training and
serving computational cost.
As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
when pre-trained on large datasets, or with modern regularization schemes,
the MLP-Mixer attains competitive scores to state-of-the-art models.
You can obtain better results by increasing the embedding dimensions,
increasing, increasing the number of mixer blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
"""


def get_mm_model():
    print("Getting MLP-Mixer model...")
    mlpmixer_blocks = keras.Sequential(
        [MLPMixerLayer(hyperparameters["num_patches"], hyperparameters["embedding_dim"], hyperparameters["dropout_rate"])
         for _ in range(hyperparameters["num_blocks"])]
    )
    mlpmixer_classifier = build_classifier(mlpmixer_blocks)
    return compile_model(mlpmixer_classifier)
