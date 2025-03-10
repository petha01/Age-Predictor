import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, sampler, Dataset
from PIL import Image
import torchvision.transforms as T

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, target_column, transform):
        self.dataframe = dataframe
        self.target_column = target_column
        self.features = dataframe['image_path'].values
        self.labels = dataframe[target_column].values
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        feature = Image.open(self.features[index]).convert('RGB')
        feature = self.transform(feature)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return feature, label

def get_utkface_loaders(target_column, batch_size=64, num_train=10000, num_valid=2000, num_test=2000):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        # T.RandomCrop(200, padding=10),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    BASE_DIR = './dataset'

    image_paths = []
    age_labels = []
    age_class_labels = []

    for filename in tqdm(os.listdir(BASE_DIR)):
        image_path = os.path.join(BASE_DIR, filename)
        temp = filename.split('_')
        age = int(temp[0])
        age_class = age // 5

        image_paths.append(image_path)
        age_labels.append(age)
        age_class_labels.append(age_class)

    data = {
            'image_path': image_paths,
            'age': age_labels,
            'age_class': age_class
        }
    df = pd.DataFrame(data)
    # Remove anything at age 100 or greater
    df = df[df['age'] < 100]
    dataset = DataFrameDataset(df, target_column, transform)
    if (num_train + num_valid + num_test) > len(dataset):
        raise ValueError("Number of train and validation samples exceeds total number of samples")

    loader_train = DataLoader(
        dataset, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(num_train))
    )

    loader_val = DataLoader(
        dataset, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train+num_valid))
    )

    loader_test = DataLoader(
        dataset,
        batch_size=batch_size, num_workers=2, shuffle=False,
        sampler=sampler.SubsetRandomSampler(range(num_train+num_valid, num_train+num_valid+num_test))
    )

    return loader_train, loader_val, loader_test