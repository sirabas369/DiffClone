"""
If you are contributing a new image encoder, implement your 
model & transforms loading functions here. 

Next, update your model info in vision/__init__.py.
"""
import sys, os, inspect
import toto_benchmark
from pathlib import Path
import torchvision.transforms as T
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn.modules.linear import Identity

_resnet_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])

def _load_model(config):
    model, embedding_dim = load_custom_model(os.path.join(os.path.dirname(inspect.getsourcefile(lambda:0)), config.agent.vision_model_path))
    model.eval()
    return model

def _load_transforms(config):
    return _resnet_transforms


device = 'cuda:0'

def load_custom_model(checkpoint_path):
    print("custom model loading")
    print(os.path.abspath(checkpoint_path))
    model = models.resnet50(pretrained=False, progress=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda:0"))
    state_dict = checkpoint
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),)

    m, u = model.load_state_dict(state_dict, strict=False)
    print("missing : \n")
    print(m)
    print("unexpected : \n")
    print(u)
    return model, 2048

# def load_custom_model(checkpoint_path):
#     print("custom model loading")
#     print(os.path.abspath(checkpoint_path))
#     model = models.resnet50(pretrained=False, progress=False)
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     # rename moco pre-trained keys
#     state_dict = checkpoint['state_dict']
#     for k in list(state_dict.keys()):
#         # retain only encoder_q up to before the embedding layer
#         if k.startswith('module.encoder_q'
#                         ) and not k.startswith('module.encoder_q.fc'):
#             # remove prefix
#             state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#         # delete renamed or unused k
#         del state_dict[k]
#     msg = model.load_state_dict(state_dict, strict=False)
#     assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
#     model.fc = Identity()
#     return model, 2048
