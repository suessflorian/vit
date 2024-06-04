import os
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import ImageClassification
# from torchvision.transforms import v2
import torchvision
from typing import Tuple

DATA = os.path.join(os.path.dirname(__file__), "data")

def cifar100(preprocess: ImageClassification, batch_size: int, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    num_workers = 1
    if device == "cuda":
        num_workers = 4

    train_transforms = torchvision.transforms.Compose([
        preprocess,
        # v2.RandAugment(),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=DATA, train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR100(root=DATA, train=False, download=True, transform=preprocess)


    if device == "cuda":
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, pin_memory_device=device)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, pin_memory_device=device)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
