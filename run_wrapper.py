#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
Distributed multimodal inference wrapper script.

Runs real-time inference over a test set of videos using configurable wrapper variants
('SotA', 'PaMo', 'ToMo') and audio SNR conditions. Loads pre-extracted audio and video packets,
simulates transmission delays, performs synchronized inference with fallback embeddings,
and collects multilabel accuracy and transmission statistics.

Main components:
    - Configuration builder that sets modality parameters, transmission window intervals (TWIs), and device.
    - Test dataset loading and packet stream initialization.
    - Audio and video feature extraction pipeline loading.
    - Model loading and fallback embedding preparation.
    - Wrapper instance that runs non-blocking multimodal inference with pre- and post-processing.
    - Aggregation of accuracy and transmission results over all test videos.
    - Results saving as compressed pickle file for downstream analysis.

Outputs:
    - Compressed pickle file saved under 'data/results/' named by run variant and SNR, containing:
        * Hamming accuracy and subset accuracy results over time windows.
        * Received packet counts per modality.
        * Tokens processed, missed packets, and number of inference windows.
        * Metadata including chosen TWI duration and stop time.

Usage:
    Run the script specifying wrapper variant and audio SNR, e.g.:
    python run_wrapper.py --variant PaMo --audio_snr_dB 1.1888

Author: Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
Date: 2025-11-11
"""



import gzip
import os
import pickle
from typing import List
from collections import deque

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fractions import Fraction

from src.datasets import PerVideoMultimodalDatasetLabels

from src.pipeline_audio import AudioPipeline
from src.pipeline_video import VideoPipeline
from src.models import PerVideoBiLSTMMultimodalClassifier

from src.packets import load_packets
from src.communication import simulate_transmission

from src.wrapper import Wrapper

from src.utils import extract_fallback_audio_token_emb

import argparse


parser = argparse.ArgumentParser(
    description="Wrapper for real-time inference in distributed multimodal systems."
)

parser.add_argument(
    "--variant",
    type=str,
    required=False,
    choices=["SotA", "PaMo", "ToMo"],
    default='PaMo',
    help="Specify the TWI-optimized variant to run: 'SotA', 'PaMo' or 'ToMo'. Default is 'PaMo'."
)

parser.add_argument(
    "--audio_snr_dB",
    type=str,
    required=False,
    choices=["-5", "1.1888", "2"],
    default='1.1888',
    help="Define the SNR for audio in decibels. Choices are: -5, 1.1888, 2 [dB]. Default is 1.1888 dB."
)

args = parser.parse_args()


# ---- Config ----
def get_config(variant: str, audio_snr_dB: str) -> dict:
    """
    Construct configuration dictionary based on variant and audio SNR.

    Args:
        variant (str): Variant name. Choices: 'SotA', 'PaMo', 'ToMo'.
        audio_snr_dB (str): Audio signal-to-noise ratio in decibels as string. Choices: '-5', '1.1888', '2'.

    Returns:
        dict: Configuration dictionary with modality and global parameters, including device setup.

    Raises:
        ValueError: If an invalid variant or SNR value is provided.

    Notes:
        The 'twi_duration' and 'stop_time' parameters are derived from empirical transmission times per variant/SNR.
        The device is set automatically based on available hardware ('cuda', 'mps', or 'cpu').
    """
    config = {
        'modalities': {},
        'global': {
            'variant': variant,
            'twi_duration': None,
            'stop_time': None, 
            'num_tokens': 10,
            'num_classes': 29,
            'token_duration': 1.0,              # seconds
            'annotations_csv': 'data/annotations.csv',
            'model_checkpoint': 'models/classification/per_video/shallow_classifier_multimodal_features_base.pth',
            'save_dir': 'data/results/',
            'device': None
        }
    }

    config['modalities']['audio'] = {
            'snr_dB': float(args.audio_snr_dB),
            'bandwidth': 1.08e6,                # Hz
            'outage_proba': 0.5,
            'packet_size': 5120,
            'packet_duration': Fraction(1, 50),
            'sampling_rate': 16000,             # Hz
    }
    
    config['modalities']['video'] = {
            'snr_dB': 0,
            'bandwidth': 100e6,                 # Hz
            'outage_proba': 0.5,
            'packet_size': 1204224,
            'packet_duration': Fraction(1, 16),
            'sampling_rate': 16,                # Hz (fps)
    }

    # Define Wrapper's TWI duration and stop time based on variant and SNR:
    # All the values are in seconds and were obtained by analyzing the distribution of the minimum between the total transmission time per modality. 
    # These values are specific to the chosen SNR and variant.
    avg_total_tx_time_video = 5.0724
    if audio_snr_dB == '-5':
        avg_total_tx_time_audio = 16.5803
    elif audio_snr_dB == '1.1888':
        avg_total_tx_time_audio = 5.0724
    elif audio_snr_dB == '2':
        avg_total_tx_time_audio = 4.4331
    else:
        raise ValueError("Invalid SNR value.")

    if variant == 'PaMo':
        twi_duration = float(np.max(((avg_total_tx_time_audio / 500), (avg_total_tx_time_video / 160)))) 
    elif variant == 'ToMo': 
        twi_duration = float(np.max((50 * (avg_total_tx_time_audio / 500), 16 * (avg_total_tx_time_video / 160))))
    elif variant == 'SotA':
        twi_duration = None
    else:
        raise ValueError("Invalid variant specified.")

    # Define stop time based on SNR
    if audio_snr_dB == '1.1888': 
        stop_time = 4.9422
    else:
        stop_time = float(np.min((avg_total_tx_time_audio, avg_total_tx_time_video))) 

    if variant == 'SotA':
        twi_duration = stop_time 

    # Round to 4 decimal places
    stop_time = float((np.ceil(stop_time * 10000) / 10000).item())
    twi_duration = float((np.ceil(twi_duration * 10000) / 10000).item())

    # Save in config
    config['global']['twi_duration'] = twi_duration
    config['global']['stop_time'] = stop_time

    # Get device
    config['device'] = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
        else torch.device('cpu')
    )

    return config


# ---- Main -----
def main(variant: str, audio_snr_dB: str) -> dict:
    """
    Main execution function for the distributed multimodal inference wrapper.

    Args:
        variant (str): Variant to run, e.g., 'SotA', 'PaMo', 'ToMo'.
        audio_snr_dB (str): Audio signal-to-noise ratio in decibels as string.

    Returns:
        dict: Aggregated inference results and metrics for all test videos.

    Side Effects:
        Loads dataset and models, runs inference over multiple test videos,
        saves results to disk as compressed pickle file.
    """
    # Get configuration
    config = get_config(variant, audio_snr_dB)

    # Logging 
    print(f"[INFO] Variant: {variant}")
    print(f"[INFO] Video SNR: {config['modalities']['video']['snr_dB']} [dB]")
    print(f"[INFO] Audio SNR: {config['modalities']['audio']['snr_dB']} [dB]")
    print(f"[INFO] TWI duration: {float(config['global']['twi_duration']):.4f} [s]")
    print(f"[INFO] Stop time: {float(config['global']['stop_time']):.4f} [s]")

    # Load test dataset
    test_ds = PerVideoMultimodalDatasetLabels(
        annotations_file=config['global']['annotations_csv'],
        split='test',
        num_tokens=config['global']['num_tokens'],
        num_classes=config['global']['num_classes']
    )
    test_videos = test_ds.video_ids

    # Get number of test videos
    num_test_videos = len(test_videos)

    # Load audio and video feature extraction pipelines
    pipeline_audio = AudioPipeline(device=config['device'])
    pipeline_audio.eval()

    pipeline_video = VideoPipeline(device='cpu', preprocess=True)
    pipeline_video.eval()

    # Fallback embedding for missing audio token
    fallback_audio_token_emb = extract_fallback_audio_token_emb(
        pipeline=pipeline_audio,
        sample_rate=config['modalities']['audio']['sampling_rate'],
        token_duration=config['global']['token_duration'],
        device=config['device']
    )
    
    # Fallback embedding for missing video sample
    dark_image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    with torch.no_grad():
        feat = pipeline_video(dark_image, return_feats=True)['feats']
    fallback_video_token_emb = feat.squeeze(0).numpy().astype(np.float32)  
    
    # Load inference model
    model = PerVideoBiLSTMMultimodalClassifier().to(config['device'])
    model.load_state_dict(torch.load(config['global']['model_checkpoint'], map_location=config['device']))
    model.eval()

    # Prepare to save results    
    results = {
        'hamming_acc': [[] for _ in range(num_test_videos)],
        'subset_acc': [[] for _ in range(num_test_videos)],
        'num_rx_pkts': {m: [[] for _ in range(num_test_videos)] for m in config['modalities'].keys()},
        'curr_num_done_tokens': {m: [[] for _ in range(num_test_videos)] for m in config['modalities'].keys()},
        'num_missed_pkts': {m: [None for _ in range(num_test_videos)] for m in config['modalities'].keys()},
        'num_twis': [None for _ in range(num_test_videos)]
    }

    # Go over test videos
    for idx, video_id in tqdm(enumerate(test_ds.video_ids), total=len(test_ds.video_ids), desc=f"Wrapper: {variant}", ascii=True):
        
        # Load packets
        stream_audio = load_packets('audio', f"data/packets/audio/{video_id}.pkl.gz")
        stream_video = load_packets('video', f"data/packets/video/{video_id}.pkl.gz")

        # Simulate transmission
        received_audio = simulate_transmission(stream_audio, config, modality='audio')
        received_video = simulate_transmission(stream_video, config, modality='video')

        received_streams = {
            'audio': deque(received_audio),
            'video': deque(received_video)
        }

        # Get ground-truth labels for the current video
        labels = test_ds[idx]['labels']

        # Create Wrapper instance
        wrapper = Wrapper(
            config=config, 
            model=model, 
            pipelines={'audio': pipeline_audio, 'video': pipeline_video},
            labels=labels, 
            received_streams=received_streams,
            fallbacks={'audio': fallback_audio_token_emb, 'video': fallback_video_token_emb},
            device=config['device']
        )

        # Run non-blocking inference
        res = wrapper.run_inference()
        
        # Aggregate results modality-wise
        results['hamming_acc'][idx].extend(res['hamming_acc'])
        results['subset_acc'][idx].extend(res['subset_acc'])

        for m in ['audio', 'video']:
            results['num_rx_pkts'][m][idx].extend(res['num_rx_pkts'][m])
            results['curr_num_done_tokens'][m][idx].extend(res['curr_num_done_tokens'][m])
            results['num_missed_pkts'][m][idx] = res['num_missed_pkts'][m]  # if single scalar per video

        results['num_twis'][idx] = res['num_twis']

    # Construct save directory and ensure it exists
    os.makedirs(config['global']['save_dir'], exist_ok=True)

    # Create SAVE_PATH dynamically based on variant and audio SNR to match config
    SAVE_PATH = os.path.join(
        config['global']['save_dir'],
        f"{config['global']['variant']}_SNR_{(config['modalities']['audio']['snr_dB'])}.pkl.gz"
    )

    # Save results, including metadata from config entries
    results['metadata'] = {
        'twi_duration': config['global']['twi_duration'],
        'stop_time': config['global']['stop_time']
    }

    # Save results to compressed pickle file
    with gzip.open(SAVE_PATH, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results


if __name__ == "__main__":
    results = main(variant=args.variant, audio_snr_dB=args.audio_snr_dB)