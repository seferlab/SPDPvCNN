import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import selected_model
from architectures.helpers.constants import threshold


hyperparameters = hyperparameters[selected_model]
# Runs at the end of every epoch and prints the confusion matrix


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, epoch_counter, time, y_test) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.epoch_counter = epoch_counter
        self.time = time
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.test_dataset)
        classes = np.argmax(predictions, axis=1)
        print(f"{confusion_matrix(self.y_test, classes)}")
        print(
            f"f1 score: {f1_score(self.y_test, classes, average='weighted')}")
        self.model.evaluate(self.test_dataset)
        print(f"\n\n")
        export_path_keras = ""
        if selected_model == "convmixer":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-{hyperparameters['filters']}x{hyperparameters['depth']}-k{hyperparameters['kernel_size']}p{hyperparameters['patch_size']}-e{self.epoch_counter}.h5"
            self.model.inner_model.save_weights(export_path_keras)
        elif selected_model == "convmixer_tf":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-{hyperparameters['filters']}x{hyperparameters['depth']}-k{hyperparameters['kernel_size']}p{hyperparameters['patch_size']}-e{self.epoch_counter}.h5"
        elif selected_model == "vision_transformer":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-tl{hyperparameters['transformer_layers']}-pd{hyperparameters['projection_dim']}-p{hyperparameters['patch_size']}-e{self.epoch_counter}.h5"
            self.model.inner_model.save_weights(export_path_keras)
            # self.model.save_weights(export_path_keras)
        elif selected_model == "mlp_mixer":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-ed{hyperparameters['embedding_dim']}-nb{hyperparameters['num_blocks']}-p{hyperparameters['patch_size']}-e{self.epoch_counter}.h5"
        elif selected_model == "cnn_ta":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-fdr{hyperparameters['first_dropout_rate']}-sdr{hyperparameters['second_dropout_rate']}-k{hyperparameters['kernel_size']}-e{self.epoch_counter}.h5"
            self.model.save_weights(export_path_keras)
        elif selected_model == "vit":
            export_path_keras = f"saved_models/{selected_model}/{threshold}/{int(self.time)}-tl{hyperparameters['transformer_layers']}-pd{hyperparameters['projection_dim']}-p{hyperparameters['patch_size']}-e{self.epoch_counter}.h5"
        self.epoch_counter += 1
