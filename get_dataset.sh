#!/bin/bash

# Download the dataset using curl with -L flag to follow redirects
curl -L -o utkface-new.zip \
  https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new

# Create a dataset directory
mkdir -p dataset

# Unzip the dataset into the dataset directory
unzip utkface-new.zip -d dataset/

# Remove the zip file after extraction
rm -f utkface-new.zip

echo "Dataset setup complete."