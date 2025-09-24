import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

import torchvision
import torchvision.transforms as transforms


def duplicate_channels(img: Tensor):
    return img.expand(3, -1, -1)


def rescaling(img: Tensor):
    return img * 2 - 1 # [-1, 1]


def load_MNIST(data_transform, train=True):
    return torchvision.datasets.MNIST(
        "./data/",
        download=True,
        train=train,
        transform=data_transform,
    )
    
    
def load_transformed_MNIST(img_size, *args, **kwargs):
    img_size = _pair(img_size)
    data_transforms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),  # Scales between [0, 1]
        transforms.Lambda(rescaling), # Scales between [-1, 1]
        transforms.Lambda(duplicate_channels),
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_MNIST(data_transform, train=True)
    test_set = load_MNIST(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    return data