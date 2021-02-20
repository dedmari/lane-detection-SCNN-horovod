#!/bin/sh

## This script downloads TuSimple data and creates segmentation labels by using gen_seg_label.py ##

# Create data path parent directory
mkdir -p "/mnt/tusimple_data/"

# Download TuSimple training data
cd /mnt/tusimple_data/
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip

# Extract training data
tar -xvf train_set.zip

# Delete zip file
rm -f train_set.zip

# Download test_label.json inside train_set
cd train_set
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json

# Generate segmentation labels
cd /workdir
python gen_seg_label.py
