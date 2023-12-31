import numpy as np
from PIL import Image
from toto_benchmark.vision.pvr_model_loading import MODEL_LIST 

EMBEDDING_DIMS = {
    'moco_conv3': 2156,
    'moco_conv5': 2048,
    'moco_conv5_robocloud': 2048,
    'r3m': 2048,
    'clip': 512,
    'resnet50': 2048,
    'byol': 512,

    'Deltas_Forced_MOCO': 2048
}

def preprocess_image(image, transforms):
    if type(image) == np.ndarray:
        assert len(image.shape) == 3
        image = Image.fromarray(image)
    processed_image = transforms(image)
    return processed_image

def load_model(config):
    model_type = config.agent.vision_model
    if model_type in MODEL_LIST:
        from .PVR import _load_model
    if model_type in ['byol_scoop', 'byol_pour']:
        from .BYOL import _load_model
    if model_type == 'resnet':
        from .Resnet import _load_model

    if model_type == 'Deltas_Forced_MOCO':
        from .Deltas_Forced_MOCO import _load_model

    return _load_model(config)

def load_transforms(config):
    model_type = config.agent.vision_model
    if model_type in MODEL_LIST:
        from .PVR import _load_transforms
    if model_type in ['byol_scoop', 'byol_pour']:
        from .BYOL import _load_transforms
    if model_type == 'resnet':
        from .Resnet import _load_transforms

    if model_type == 'Deltas_Forced_MOCO':
        from .Deltas_Forced_MOCO import _load_transforms

    return _load_transforms(config)