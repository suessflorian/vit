import torchvision
import torch
from torchvision.models import VisionTransformer
from typing import Tuple

from torchvision.models.vision_transformer import ImageClassification

CIFAR_100_CLASSES = 100

def transformer() -> Tuple[VisionTransformer, ImageClassification]:
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    # honed for classes of the CIFAR100 dataset
    model.heads = torch.nn.Sequential(torch.nn.Linear(model.hidden_dim, CIFAR_100_CLASSES))
    return model, weights.transforms()
