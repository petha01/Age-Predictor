#!/bin/bash

# Download the dataset using curl with -L flag to follow redirects
curl -L -o utkface-new.zip \
  https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new

# Unzip the dataset into the dataset directory
unzip utkface-new.zip -d ./

# Remove the zip file after extraction
mv UTKFace dataset
rm -f utkface-new.zip
rm -rf crop_part1
rm -rf utkface_aligned_cropped

echo "Dataset setup complete."