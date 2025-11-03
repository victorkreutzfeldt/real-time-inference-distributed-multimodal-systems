#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import pickle
import time

import numpy as np
import torch
import av
import h5py
from tqdm import tqdm
import imageio

from src._class_pipeline_video import VideoPipeline
from src.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# ====================== Config ======================
ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'

VIDEO_FPS = 16 # video frames per second
NUM_CHUNKS = 10 # number of chunks to split audio into
CHUNK_DURATION = 1.0  # 1 second per chunk
NUM_FRAMES_PER_CHUNK = int(VIDEO_FPS * CHUNK_DURATION)  # number of frames per chunk
NUM_EXPECTED_VIDEO_FRAMES = int(VIDEO_FPS * CHUNK_DURATION * NUM_CHUNKS) # expected number of frames for the entire video

# Define paths for output
OUTPUT_IMAGES_PATH = 'data/raw/video_imgs.h5'

OUTPUT_FEATURES_PATH = 'data/classification/features/video.h5'
OUTPUT_FEATURES_TEM_GPA_PATH = 'data/classification/features/video_tem_gpa.h5'
OUTPUT_FEATURES_SPA_GPA_PATH = 'data/classification/features/video_spa_gpa.h5'
OUTPUT_FEATURES_TEM_SPA_GPA_PATH = 'data/classification/features/video_tem_spa_gpa.h5'

OUTPUT_TIMING_PATH = 'data/classification/timing/video_pipeline.pth'

# Check if output directories exist, if not create them
os.makedirs(os.path.dirname(OUTPUT_IMAGES_PATH), exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FEATURES_TEM_GPA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FEATURES_SPA_GPA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FEATURES_TEM_SPA_GPA_PATH), exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_TIMING_PATH), exist_ok=True)

# Check available device
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    else torch.device('cpu')
)

if DEVICE.type == 'cuda':
    NUM_CPUS = 2
    PIN_MEMORY = True
else:
    NUM_CPUS = 0
    PIN_MEMORY = False

print(f"[INFO] Using device: {DEVICE}")

# ====================== Main ======================
if __name__ == "__main__":

    # Load video pipeline
    video_pipeline = VideoPipeline(pretrained=True, device=DEVICE, preprocess=True, postprocess=False)
    video_pipeline.eval()
    
    # Get video files
    video_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.avi')]

    # Save timing
    timing = {'image': [], 'processing': []}    

    # ====================== Feature Extraction ======================
    # Open HDF5 file once and save per video
    with h5py.File(OUTPUT_IMAGES_PATH, 'w') as hf_imgs,\
        h5py.File(OUTPUT_FEATURES_PATH, 'w') as hf_feats,\
        h5py.File(OUTPUT_FEATURES_TEM_GPA_PATH, 'w') as hf_feats_tem_gpa,\
        h5py.File(OUTPUT_FEATURES_SPA_GPA_PATH, 'w') as hf_feats_spa_gpa,\
        h5py.File(OUTPUT_FEATURES_TEM_SPA_GPA_PATH, 'w') as hf_feats_tem_spa_gpa: 

            # Iterate over video files
            for name in tqdm(video_files, desc="Processing videos", ascii=True):

                # Get file path
                video_path = os.path.join(INPUT_PATH, name)
                
                # Get video ID from filename
                video_id = os.path.splitext(name)[0]

                # Read all frames once
                time_start = time.time()
                
                video = imageio.get_reader(video_path, 'ffmpeg')
                imgs = [img for img in video]
                video.close()

                time_end = time.time()
                timing['image'].append((time_end - time_start) / NUM_CHUNKS)

                # Convert imgs to numpy array
                imgs = np.array(imgs, dtype=np.uint8)
                total_num_imgs = len(imgs)

                # Assert total number of imgs
                if total_num_imgs > NUM_EXPECTED_VIDEO_FRAMES:
                    
                    # Truncate if it is longer than expected
                    imgs = imgs[:NUM_EXPECTED_VIDEO_FRAMES]
                
                # Pad if it is shorter than expected
                if total_num_imgs < NUM_EXPECTED_VIDEO_FRAMES:

                    # Calculate number of imgs to pad
                    num_to_pad = NUM_EXPECTED_VIDEO_FRAMES - total_num_imgs
                    
                    # 'imgs' is of shape (total_num_imgs, H, W, C)
                    image_shape = imgs.shape[1:]  # (H, W, C)
                    
                    # Create zero imgs to pad
                    zero_imgs = np.zeros((num_to_pad, *image_shape), dtype=imgs.dtype)
                    
                    # Concatenate along the batch dimension (axis=0)
                    imgs = np.concatenate((imgs, zero_imgs), axis=0)

                # Extract features
                with torch.no_grad():

                    time_start = time.time()
                    out = video_pipeline(imgs, return_feats=True, return_embs=False)
                    time_end = time.time()

                    timing['processing'].append((time_end - time_start) / NUM_EXPECTED_VIDEO_FRAMES)
        
                # Get features and convert to numpy
                feats = out['feats'].cpu().numpy().astype(np.float32)

                # Reshape to (NUM_CHUNKS, NUM_FRAMES_PER_CHUNK, C, H, W)
                imgs = imgs.reshape(NUM_CHUNKS, NUM_FRAMES_PER_CHUNK, *imgs.shape[1:])
                feats = feats.reshape(NUM_CHUNKS, NUM_FRAMES_PER_CHUNK, *feats.shape[1:])

                # Calculate GPAs
                feats_tem_gpa = feats.mean(axis=1)
                feats_spa_gpa = feats.mean(axis=(-1, -2))
                feats_tem_spa_gpa = feats_spa_gpa.mean(axis=1)
    
                # Save to output files
                hf_imgs.create_dataset(video_id, data=imgs, dtype=np.uint8, compression="gzip")
                
                hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
                hf_feats_tem_gpa.create_dataset(video_id, data=feats_tem_gpa, dtype=np.float32, compression="gzip")
                hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")
                hf_feats_tem_spa_gpa.create_dataset(video_id, data=feats_tem_spa_gpa, dtype=np.float32, compression="gzip")

    # Save timing information
    with open(OUTPUT_TIMING_PATH, 'wb') as f:
        pickle.dump(timing, f)
