#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
Multimodal Packet Extraction Script

This script extracts audio and video packets from input video files for a multimodal dataset.
It uses FFmpeg to extract frames and audio, processes audio into fixed-size packets,
stores packets as serialized objects including metadata, and saves them for downstream processing.

It handles:
    - Audio extraction at 16 kHz mono, chunked into fixed sample windows.
    - Video frame extraction at 16 fps, resized to 224x224 pixels.
    - Packet metadata capturing presentation timestamps, durations, sizes, and payloads.
    - Temporary file management and cleanup.
    - Processing of a dataset subset (e.g., test split) based on provided annotations.

Outputs:
    - Audio packets saved as gzipped pickle files in 'data/packets/audio/<video_id>.pkl.gz', each containing
      Packet objects with metadata and payloads (audio chunks).
    - Video packets saved as gzipped pickle files in 'data/packets/video/<video_id>.pkl.gz', each containing
      Packet objects with metadata and payloads (frames or extracted features).
    - Temporary files and directories cleaned after processing.

Usage:
Simply run:
`python packetization.py`

@author Victor Kreützfeldt (@victorkreutzfeldt)
@date 2025-11-11
"""

import logging

import numpy as np 
import pandas as pd
import h5py
from tqdm import tqdm

from src.packetization import Packetization


def main(config):

    # Create a logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get test videos
    annotations = pd.read_csv(config['global']['annotations_csv_path'])
    test_video_ids = annotations[annotations['split'] == 'test']['video_id'].unique().tolist()
    num_test_videos = len(test_video_ids)

    # Instantiate the packetization process
    packetization = Packetization(config)

    # Prepare to save results
    results = {}
    results['packet_counts'] = {m: [None for _ in range(num_test_videos)] for m in config['modalities'].keys()}
   
    # Instead of saving raw images, we save video representations as payloads for each video packet
    representations_path = config['pipelines']['video']['representations_path']

    with h5py.File(representations_path, 'r') as video_h5:

        # Iterate over test video IDs with a progress bar
        for vv, vid in tqdm(enumerate(test_video_ids), desc="Processing test videos", ascii=True, total=len(test_video_ids)):

            # Load video representations for the current video
            video_payloads = np.array(video_h5[vid])

            # Packetize the current video representations
            res = packetization.packetize_single_video(vid, video_payloads)

            # Store packet counts per modality for this video
            for modality in config['modalities'].keys():
                results['packet_counts'][modality][vv] = res['packet_counts'][modality]
                 
    # Log summary of packet counts per video
    for vv in range(len(test_video_ids)):
        vid = test_video_ids[vv]
        logger.info(f"{vid}: Audio packets={results['packet_counts']['audio'][vv]}, Video packets={results['packet_counts']['video'][vv]}")

    # Log average packet counts over all test videos
    logger.info("\nAverage packet counts:")
    logger.info(f"Average Audio Packets: {np.mean(results['packet_counts']['audio']):.2f}")
    logger.info(f"Average Video Packets: {np.mean(results['packet_counts']['video']):.2f}")

if __name__ == '__main__':

    # Create a config dictionary
    config = {
        'global': {
            'video_duration': 10.0, # seconds
            'annotations_csv_path': 'data/annotations.csv',
            'videos_dir': 'data/AVE_trimmed'
        },
        'modalities': {},
        'pipelines': {},        
    }

    # Audio configurations
    config['modalities']['audio'] = {
        'sampling_rate': 16000,  # Hz
        'packet_size_samples': 320, 
        'num_channels': 1,
        'resolution': None,
        'tmp_dir': 'data/_tmp/audio',
        'packets_dir': 'data/packets/audio'
    }

    # Video configurations
    config['modalities']['video'] = {
        'sampling_rate': 16,    # Hz
        'packet_size_samples': 1, 
        'num_channels': 3,
        'resolution': (224, 224),
        'tmp_dir': 'data/_tmp/video',
        'packets_dir': 'data/packets/video'
    }
    
    # Pipeline configurations
    config['pipelines']['video'] = {
        'representations_path': 'data/representations/features/video.h5'
    }

    main(config)