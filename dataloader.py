import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T

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

# Fix this