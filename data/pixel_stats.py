# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: data/pixel_stats.py

import torch
from torchvision import transforms

import h5py
import pandas as pd

from tqdm import tqdm

# H5 file containing images
ANNOTATIONS_FILE = "data/annotations.csv"
VIDEO_H5_FILE = "data/raw/video_imgs.h5"

# Open file
video_h5 = h5py.File(VIDEO_H5_FILE, 'r') 

# Load and filter by split
annotations = pd.read_csv(ANNOTATIONS_FILE)
annotations = annotations[annotations['split'] == 'train'].reset_index(drop=True)

# Unique video IDs in this split
video_ids = annotations['video_id'].unique()

# Define number of tokens
num_tokens = 10

# Transformation
transform = transforms.ToTensor()  # only ToTensor; no Normalize here!

# Initialize counters
n_images = 0
total_sum = torch.zeros(3)
total_sum_sq = torch.zeros(3)
total_pixels = 0

# Go over training videos
for video_id in tqdm(video_ids, desc="Going through training videos"):

    # Go over tokens
    for tt in range(num_tokens):
        imgs = video_h5[video_id][tt]
        
        for ii in range(16):
            img = imgs[ii]
            tensor = transform(img)  # shape: [3, H, W]
            n_images += 1
            total_pixels += tensor.shape[1] * tensor.shape[1]
            total_sum += tensor.sum(dim=(1, 2))
            total_sum_sq += (tensor ** 2).sum(dim=(1, 2))

mean = total_sum / total_pixels
std = (total_sum_sq / total_pixels - mean ** 2).sqrt()

print('Mean per channel:', mean.tolist())
print('Std per channel:', std.tolist())
