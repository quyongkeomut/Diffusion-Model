import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

import torchvision
import torchvision.transforms as transforms

        
def duplicate_channels(img: Tensor):
    return img.expand(3, -1, -1)


def rescaling(img: Tensor):
    return img * 2 - 1 # [-1, 1]


def load_FER2013(data_transform, train: bool = True):
    return torchvision.datasets.FER2013(
        "./data/",
        split='train' if train is True else 'test',
        transform=data_transform,
    )
    

def load_transformed_FER2013(img_size, *args, **kwargs):
    img_size = _pair(img_size)
    data_transforms = [
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0, 1]
        transforms.Lambda(rescaling), # Scales between [-1, 1]
        transforms.Lambda(duplicate_channels),
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_FER2013(data_transform, train=True)
    test_set = load_FER2013(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    return data


if __name__ == '__main__':
    data = load_transformed_FER2013(64)
    for i, instance in enumerate(data):
        img, label = instance 
        if img is None or label is None:
            if img is None:
                print('none image detected')
            if label is None:
                print('none label detected')