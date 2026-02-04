from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_fashion_mnist(normal_class):
    transform = transforms.ToTensor()

    train = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    normal_indices = [
        i for i, (_, y) in enumerate(train) if y == normal_class
    ]

    return Subset(train, normal_indices), test

