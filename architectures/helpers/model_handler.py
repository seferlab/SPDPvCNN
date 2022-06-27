from architectures.helpers.constants import selected_model
from architectures.convmixer_tf import get_cm_tf_model
from architectures.convmixer import get_cm_model
from architectures.vision_transformer import get_vit_model
from architectures.mlp_mixer import get_mm_model
from architectures.cnn_ta import get_ct_model
from architectures.vit import get_vit


def get_model():
    if selected_model == "convmixer":
        return get_cm_model()
    elif selected_model == "convmixer_tf":
        return get_cm_tf_model()
    elif selected_model == "vision_transformer":
        return get_vit_model()
    elif selected_model == "mlp_mixer":
        return get_mm_model()
    elif selected_model == "cnn_ta":
        return get_ct_model()
    elif selected_model == "vit":
        return get_vit()
