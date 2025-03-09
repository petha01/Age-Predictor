import torch
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader, sampler

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    # T.RandomCrop(200, padding=10),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_utkface_loaders(dataset_path="./dataset", batch_size=64, num_train=30000, num_valid=5000):
    utkface_dataset = dset.ImageFolder(dataset_path, transform=train_transform)
    if (num_train + num_valid) > len(utkface_dataset):
        raise ValueError("Number of train and validation samples exceeds total number of samples")

    loader_train = DataLoader(
        utkface_dataset, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(num_train))
    )

    loader_val = DataLoader(
        utkface_dataset, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train+num_valid))
    )

    loader_test = DataLoader(
        dset.ImageFolder(dataset_path, transform=test_transform),
        batch_size=batch_size, num_workers=2, shuffle=False
    )

    return loader_train, loader_val, loader_test