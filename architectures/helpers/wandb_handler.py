import wandb
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import selected_model
from architectures.helpers.constants import threshold

hyperparameters = hyperparameters[selected_model]


def initialize_wandb():
    if selected_model == "convmixer":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "filters": hyperparameters["filters"],
                       "depth": hyperparameters["depth"],
                       "kernel_size": hyperparameters["kernel_size"],
                       "patch_size": hyperparameters["patch_size"],
                       "threshold": f"0.{threshold}",
                       "image_size": hyperparameters["image_size"]
                   })
    elif selected_model == "convmixer_tf":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "filters": hyperparameters["filters"],
                       "depth": hyperparameters["depth"],
                       "kernel_size": hyperparameters["kernel_size"],
                       "patch_size": hyperparameters["patch_size"],
                       "threshold": f"0.{threshold}",
                       "image_size": hyperparameters["image_size"]
                   })
    elif selected_model == "vision_transformer":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "mlp_mixer":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "dropout_rate": hyperparameters["dropout_rate"],
                       "embedding_dim": hyperparameters["embedding_dim"],
                       "patch_size": hyperparameters["patch_size"],
                       "num_blocks": hyperparameters["num_blocks"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "cnn_ta":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "image_size": hyperparameters["image_size"],
                       "first_dropout_rate": hyperparameters["first_dropout_rate"],
                       "second_dropout_rate": hyperparameters["second_dropout_rate"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "vit":
        wandb.init(project=f"{selected_model}", entity="spdpvcnn",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "layer_norm_eps": hyperparameters["layer_norm_eps"],
                       "threshold": f"0.{threshold}",
                   })
