#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video feature extraction pipeline script.

This script reads video files, extracts frames, splits videos into fixed-length tokens, preprocesses frames,
and passes them through a pretrained VideoPipeline to extract deep features. It computes temporal, spatial,
and spatio-temporal global pooling aggregations (GPAs) on extracted features.

The script saves intermediate frames (optionally), extracted features, pooled features, and processing
timings to HDF5 and pickle files for downstream training and evaluation.

Constants for file paths, sampling (frame) rates, and processing parameters are defined in a config
dictionary inside the main() function.

Main Components:
    - VideoPipelineWrapper: Handles loading raw video frames, padding/truncation to fixed lengths, batch
    feature extraction on GPU or CPU, optional preprocessing steps (e.g., normalization), and saving results
    in HDF5 files. It also records detailed timing information for performance analysis.
    - main: Drives batch processing over a directory of videos.

Outputs:
    - HDF5 datasets containing:
        * Raw preprocessed video frames per token (NOTE: commented out because consume too much disk space).
        * Extracted convolutional features.
        * Temporal GPA features.
        * Spatial GPA features.
        * Temporal-Spatial GPA features.
    - Pickle files with timing statistics per video processed.

Usage:
    Simply run:
    python _apply_pipeline_video.py

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import os
import pickle
import time
from typing import Tuple

import numpy as np
import torch
import h5py
import imageio
from tqdm import tqdm

from src.pipeline_video import VideoPipeline
from src.utils import set_seed


class VideoPipelineWrapper:
    """
    Wrapper for video feature extraction pipeline supporting loading, preprocessing,
    batch tokenization, feature extraction, postprocessing, and saving outputs.

    Args:
        device (torch.device): Computation device.
        pretrained (bool): Whether to load pretrained weights.
        preprocess (bool): Apply preprocessing (normalization).
        postprocess (bool): Apply postprocessing steps (currently TODO).
    """

    def __init__(self, device: torch.device, pretrained: bool = True, preprocess: bool = True, postprocess: bool = False):
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.pipeline = VideoPipeline(pretrained=pretrained, device=device, preprocess=preprocess, postprocess=postprocess)
        self.pipeline.eval()
        # Postprocessing support TODO if implemented

    def load_video_frames(self, video_path: str) -> np.ndarray:
        video = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in video]
        video.close()
        return np.array(frames, dtype=np.uint8)

    def pad_or_truncate_imgs(self, imgs: np.ndarray, expected_num_samples: int) -> np.ndarray:
        total_num_imgs = imgs.shape[0]
        if total_num_imgs > expected_num_samples:
            imgs = imgs[:expected_num_samples]
        elif total_num_imgs < expected_num_samples:
            pad_count = expected_num_samples - total_num_imgs
            zero_imgs = np.zeros((pad_count, *imgs.shape[1:]), dtype=imgs.dtype)
            imgs = np.concatenate((imgs, zero_imgs), axis=0)
        return imgs

    def compute_gpas(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats_tem_gpa = feats.mean(axis=1)
        feats_spa_gpa = feats.mean(axis=(-1, -2))
        feats_tem_spa_gpa = feats_spa_gpa.mean(axis=1)
        return feats_tem_gpa, feats_spa_gpa, feats_tem_spa_gpa

    def process_videos(self,
                       input_path: str,
                       num_tokens: int,
                       token_duration: float,
                       target_sampling_rate: int,
                       output_images_path: str,
                       output_feats_path: str,
                       output_feats_tem_gpa_path: str,
                       output_feats_spa_gpa_path: str,
                       output_feats_tem_spa_gpa_path: str,
                       output_timing_path: str):

        num_samples_per_token = int(target_sampling_rate * token_duration)
        total_expected_samples = int(target_sampling_rate * token_duration * num_tokens)

        video_files = [f for f in os.listdir(input_path) if f.endswith('.avi')]
        timing = {'image': [], 'processing': []}

        with h5py.File(output_images_path, 'w') as hf_imgs, \
            h5py.File(output_feats_path, 'w') as hf_feats, \
            h5py.File(output_feats_tem_gpa_path, 'w') as hf_feats_tem_gpa, \
            h5py.File(output_feats_spa_gpa_path, 'w') as hf_feats_spa_gpa, \
            h5py.File(output_feats_tem_spa_gpa_path, 'w') as hf_feats_tem_spa_gpa:

            for name in tqdm(video_files, desc="Processing videos", ascii=True):
                video_path = os.path.join(input_path, name)
                video_id = os.path.splitext(name)[0]

                start_time = time.time()
                imgs = self.load_video_frames(video_path)
                end_time = time.time()
                timing['image'].append((end_time - start_time) / num_tokens)

                imgs = self.pad_or_truncate_imgs(imgs, total_expected_samples)

                with torch.no_grad():
                    start_time = time.time()
                    out = self.pipeline(imgs, return_feats=True, return_embs=False)
                    end_time = time.time()
                    timing['processing'].append((end_time - start_time) / num_tokens)

                feats = out['feats'].cpu().numpy().astype(np.float32)
                feats = feats.reshape(num_tokens, num_samples_per_token, *feats.shape[1:])

                feats_tem_gpa, feats_spa_gpa, feats_tem_spa_gpa = self.compute_gpas(feats)

                # Save datasets (uncomment to save raw images)
                # hf_imgs.create_dataset(video_id, data=imgs, dtype=np.uint8, compression="gzip")

                hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
                hf_feats_tem_gpa.create_dataset(video_id, data=feats_tem_gpa, dtype=np.float32, compression="gzip")
                hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")
                hf_feats_tem_spa_gpa.create_dataset(video_id, data=feats_tem_spa_gpa, dtype=np.float32, compression="gzip")

        with open(output_timing_path, 'wb') as f:
            pickle.dump(timing, f)


def main():
    config = {
        "input_path": "data/AVE_trimmed",
        "output_images_path": "data/raw/video_imgs.h5",
        "output_feats_path": "data/classification/features/video.h5",
        "output_feats_tem_gpa_path": "data/classification/features/video_tem_gpa.h5",
        "output_feats_spa_gpa_path": "data/classification/features/video_spa_gpa.h5",
        "output_feats_tem_spa_gpa_path": "data/classification/features/video_tem_spa_gpa.h5",
        "output_timing_path": "data/classification/timing/video_pipeline.pth",
        "num_tokens": 10,
        "token_duration": 1.0,
        "target_sampling_rate": 16,
        "device": (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
            else torch.device('cpu')
        ),
        "seed": 42
    }

    set_seed(config['seed'])

    video_wrapper = VideoPipelineWrapper(
        device=config['device'],
        pretrained=True,
        preprocess=True,
        postprocess=False
    )

    video_wrapper.process_videos(
        input_path=config["input_path"],
        num_tokens=config["num_tokens"],
        token_duration=config["token_duration"],
        target_sampling_rate=config["target_sampling_rate"],
        output_images_path=config["output_images_path"],
        output_feats_path=config["output_feats_path"],
        output_feats_tem_gpa_path=config["output_feats_tem_gpa_path"],
        output_feats_spa_gpa_path=config["output_feats_spa_gpa_path"],
        output_feats_tem_spa_gpa_path=config["output_feats_tem_spa_gpa_path"],
        output_timing_path=config["output_timing_path"],
    )


if __name__ == "__main__":
    main()
