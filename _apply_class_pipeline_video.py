#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import pickle
import time
import logging

import numpy as np
import torch
import h5py
import imageio
from tqdm import tqdm

from src._class_pipeline_video import VideoPipeline
from src.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# ---- Config ----
ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'

TARGET_FPS = 16  # video frames per second
NUM_TOKENS = 10  # number of tokens to split video into
TOKEN_DURATION = 1.0  # 1 second per token
NUM_SAMPLES_PER_TOKEN = int(TARGET_FPS * TOKEN_DURATION)  # number samples per token
TARGET_NUM_SAMPLES = int(TARGET_FPS * TOKEN_DURATION * NUM_TOKENS)  # expected total samples

# Define output paths
OUTPUT_IMAGES_PATH = 'data/raw/video_imgs.h5'
OUTPUT_FEATURES_PATH = 'data/classification/features/video.h5'
OUTPUT_FEATURES_TEM_GPA_PATH = 'data/classification/features/video_tem_gpa.h5'
OUTPUT_FEATURES_SPA_GPA_PATH = 'data/classification/features/video_spa_gpa.h5'
OUTPUT_FEATURES_TEM_SPA_GPA_PATH = 'data/classification/features/video_tem_spa_gpa.h5'
OUTPUT_TIMING_PATH = 'data/classification/timing/video_pipeline.pth'

# Ensure output directories exist
output_dirs = set(os.path.dirname(p) for p in [
    OUTPUT_IMAGES_PATH, OUTPUT_FEATURES_PATH, OUTPUT_FEATURES_TEM_GPA_PATH,
    OUTPUT_FEATURES_SPA_GPA_PATH, OUTPUT_FEATURES_TEM_SPA_GPA_PATH, OUTPUT_TIMING_PATH
])
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# Setup device
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    else torch.device('cpu')
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"[INFO] Using device: {DEVICE}")


def load_video_frames(video_path: str) -> np.ndarray:
    """
    Reads all frames from a video and returns as a numpy array of shape (num_frames, H, W, C).
    """
    video = imageio.get_reader(video_path, 'ffmpeg')
    frames = [img for img in video]
    video.close()
    imgs = np.array(frames, dtype=np.uint8)
    return imgs


def pad_or_truncate_imgs(imgs: np.ndarray, expected_num_samples: int) -> np.ndarray:
    """
    Pad with zeros or truncate to expected_num_samples along axis 0.
    """
    total_num_imgs = imgs.shape[0]
    if total_num_imgs > expected_num_samples:
        imgs = imgs[:expected_num_samples]
    elif total_num_imgs < expected_num_samples:
        num_to_pad = expected_num_samples - total_num_imgs
        zero_imgs = np.zeros((num_to_pad, *imgs.shape[1:]), dtype=imgs.dtype)
        imgs = np.concatenate((imgs, zero_imgs), axis=0)
    return imgs


def compute_gpas(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute temporal, spatial, and spatio-temporal GPAs on features.
    """
    feats_tem_gpa = feats.mean(axis=1)
    feats_spa_gpa = feats.mean(axis=(-1, -2))
    feats_tem_spa_gpa = feats_spa_gpa.mean(axis=1)
    return feats_tem_gpa, feats_spa_gpa, feats_tem_spa_gpa


def main():
    video_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.avi')]
    timing = {'image': [], 'processing': []}

    video_pipeline = VideoPipeline(pretrained=True, device=DEVICE, preprocess=True, postprocess=False)
    video_pipeline.eval()

    with h5py.File(OUTPUT_IMAGES_PATH, 'w') as hf_imgs, \
         h5py.File(OUTPUT_FEATURES_PATH, 'w') as hf_feats, \
         h5py.File(OUTPUT_FEATURES_TEM_GPA_PATH, 'w') as hf_feats_tem_gpa, \
         h5py.File(OUTPUT_FEATURES_SPA_GPA_PATH, 'w') as hf_feats_spa_gpa, \
         h5py.File(OUTPUT_FEATURES_TEM_SPA_GPA_PATH, 'w') as hf_feats_tem_spa_gpa:

        for name in tqdm(video_files, desc="Processing videos", ascii=True):
            video_path = os.path.join(INPUT_PATH, name)
            video_id = os.path.splitext(name)[0]

            # Read frames once with timing
            time_start = time.time()
            imgs = load_video_frames(video_path)
            time_end = time.time()
            timing['image'].append((time_end - time_start) / NUM_TOKENS)

            imgs = pad_or_truncate_imgs(imgs, TARGET_NUM_SAMPLES)
            imgs = imgs.reshape(NUM_TOKENS, NUM_SAMPLES_PER_TOKEN, *imgs.shape[1:])

            with torch.no_grad():
                time_start = time.time()
                out = video_pipeline(imgs, return_feats=True, return_embs=False)
                time_end = time.time()
                timing['processing'].append((time_end - time_start) / NUM_TOKENS)

            feats = out['feats'].cpu().numpy().astype(np.float32)
            feats = feats.reshape(NUM_TOKENS, NUM_SAMPLES_PER_TOKEN, *feats.shape[1:])

            feats_tem_gpa, feats_spa_gpa, feats_tem_spa_gpa = compute_gpas(feats)

            # Save datasets
            
            # Uncomment if saving raw images needed
            # hf_imgs.create_dataset(video_id, data=imgs, dtype=np.uint8, compression="gzip")

            hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
            hf_feats_tem_gpa.create_dataset(video_id, data=feats_tem_gpa, dtype=np.float32, compression="gzip")
            hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")
            hf_feats_tem_spa_gpa.create_dataset(video_id, data=feats_tem_spa_gpa, dtype=np.float32, compression="gzip")

    # Save timing
    with open(OUTPUT_TIMING_PATH, 'wb') as f:
        pickle.dump(timing, f)


if __name__ == "__main__":
    main()
