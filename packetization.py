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

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import os
import shutil
import subprocess
import numpy as np
import h5py
import pickle
import gzip
from PIL import Image
from tqdm import tqdm
import pandas as pd
import scipy.io.wavfile
import logging

from fractions import Fraction
from typing import List, Tuple, Optional
from src.packets import Packet


# ---- Configuration ----
TARGET_VIDEO_DURATION = 10.0
TARGET_SAMPLING_RATE_AUDIO = 16000
TARGET_SAMPLING_RATE_VIDEO = 16
IMAGE_RESOLUTION = (224, 224)

TARGET_NUM_VIDEO_PACKETS = int(TARGET_SAMPLING_RATE_VIDEO * TARGET_VIDEO_DURATION)  # 160 visual samples
AUDIO_WINDOW_SIZE = 320 # samples per audio packet
TARGET_NUM_AUDIO_PACKETS = int(TARGET_SAMPLING_RATE_AUDIO * TARGET_VIDEO_DURATION / AUDIO_WINDOW_SIZE)

ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'
VIDEO_FEATURES_PATH = 'data/classification/features/video.h5'

OUTPUT_PATH = 'data/packets'


def ensure_dirs(dirs: List[str]) -> None:
    """Ensure that a list of directories exist, creating them if necessary."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def extract_video_frames_ffmpeg(input_path: str, tmp_dir: str) -> None:
    """
    Extract and resize video frames at target FPS using FFmpeg.

    Args:   
        input_path (str): Path to input video file.
        tmp_dir (str): Directory to save extracted frames.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    for f in os.listdir(tmp_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(tmp_dir, f))

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'quiet',
        '-i', input_path,
        '-vf', f'fps={TARGET_SAMPLING_RATE_VIDEO},scale={IMAGE_RESOLUTION[0]}:{IMAGE_RESOLUTION[1]}',
        os.path.join(tmp_dir, 'frame_%05d.png')
    ]
    subprocess.run(cmd, check=True)


def extract_audio_ffmpeg(input_path: str, tmp_wav_path: str) -> None:
    """
    Extract mono audio, downsampled to target sampling rate, as WAV using FFmpeg.

    Args:
        input_path (str): Path to input video file.
        tmp_wav_path (str): Path to output WAV file.
    """
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'quiet',
        '-i', input_path,
        '-ac', '1',
        '-ar', str(TARGET_SAMPLING_RATE_AUDIO),
        '-vn',
        tmp_wav_path
    ]
    subprocess.run(cmd, check=True)


def load_audio_packets(wav_path: str, window_size: int = AUDIO_WINDOW_SIZE, target_packets: int = TARGET_NUM_AUDIO_PACKETS) -> Tuple[np.ndarray, int]:
    """
    Load audio samples from WAV and split into fixed-size packets.

    Args:
        wav_path (str): Path to WAV audio file.
        window_size (int): Samples per packet.
        target_packets (int): Desired number of packets.

    Returns:
        packets (np.ndarray): Array of packets of shape [target_packets, window_size].
    """
    sr, waveform = scipy.io.wavfile.read(wav_path)
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0

    total_samples = window_size * target_packets
    assert sr == TARGET_SAMPLING_RATE_AUDIO, f"Expected {TARGET_SAMPLING_RATE_AUDIO}, got {sr}."
    assert waveform.ndim == 1, f"Expected mono audio, got {waveform.ndim} channels."

    if len(waveform) < total_samples:
        waveform = np.pad(waveform, (0, total_samples - len(waveform)), mode='constant')
    else:
        waveform = waveform[:total_samples]

    packets = waveform.reshape((target_packets, window_size))
    return packets


def save_audio_packets(audio_packets: np.ndarray, save_dir: str, video_id: str) -> List[Packet]:
    """
    Save audio packets as gzipped pickled Packet objects with metadata.

    Args:
        audio_packets (np.ndarray): Audio chunks (num_packets x window_size).
        save_dir (str): Directory to save packets.
        video_id (str): Video identifier.

    Returns:
        List[Packet]: Packet instances saved.
    """
    packets_path = os.path.join(save_dir, f'audio/{video_id}.pkl.gz')
    ensure_dirs([os.path.dirname(packets_path)])

    packets = []
    packet_duration = Fraction(AUDIO_WINDOW_SIZE, TARGET_SAMPLING_RATE_AUDIO)
    size_bits = AUDIO_WINDOW_SIZE * 16  # 16 bits per sample

    for idx, samples in enumerate(audio_packets):
        pkt = Packet(
            stream_type='audio',
            pts=idx,
            pts_time=packet_duration * idx,
            duration=packet_duration,
            size_bits=size_bits,
            sample_rate=TARGET_SAMPLING_RATE_AUDIO,
            time_base=Fraction(1, TARGET_SAMPLING_RATE_AUDIO),
            nb_channels=1,
            payload=samples.astype(np.float32)
        )
        packets.append(pkt)

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)

    return packets


def save_video_packets(frame_dir: str, save_dir: str, video_id: str, video_h5: Optional[h5py.File] = None) -> List[Packet]:
    """
    Save video frames as gzipped pickled Packet objects with metadata.

    Args:
        frame_dir (str): Directory with PNG frames.
        save_dir (str): Directory to save packets.
        video_id (str): Video identifier.
        video_h5 (h5py.File, optional): HDF5 file with video features for payload.

    Returns:
        List[Packet]: Packet instances saved.
    """
    packets_path = os.path.join(save_dir, f'video/{video_id}.pkl.gz')
    ensure_dirs([os.path.dirname(packets_path)])

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
    if len(frame_files) < TARGET_NUM_VIDEO_PACKETS:
        raise RuntimeError(f"Not enough frames: got {len(frame_files)}, expected {TARGET_NUM_VIDEO_PACKETS}")
    frame_files = frame_files[:TARGET_NUM_VIDEO_PACKETS]

    video_payloads = None
    if video_h5 is not None:
        video_payloads = np.array(video_h5[video_id][:]).reshape(TARGET_NUM_VIDEO_PACKETS, 512, 7, 7)

    packets = []
    packet_duration = Fraction(1, TARGET_SAMPLING_RATE_VIDEO)

    for idx, fname in enumerate(frame_files):
        path = os.path.join(frame_dir, fname)
        img = Image.open(path).convert('RGB')
        np_img = np.array(img, dtype=np.uint8)
        size_bits = np_img.nbytes * 8

        pkt = Packet(
            stream_type='video',
            pts=idx,
            pts_time=packet_duration * idx,
            duration=packet_duration,
            size_bits=size_bits,
            sample_rate=TARGET_SAMPLING_RATE_VIDEO,
            time_base=Fraction(1, TARGET_SAMPLING_RATE_VIDEO),
            nb_channels=3,
            resolution=IMAGE_RESOLUTION[::-1],
            payload=video_payloads[idx] if video_payloads is not None else np_img
        )
        packets.append(pkt)

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)

    return packets


def cleanup_temp_files(tmp_dirs: List[str], tmp_files: List[str]) -> None:
    """
    Remove temporary directories and files safely.
    """
    for f in tmp_files:
        if os.path.isfile(f):
            os.remove(f)
    for d in tmp_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def process_single_video(video_id: str, video_h5: h5py.File) -> Tuple[int, int]:
    """
    Orchestrate extraction of audio and video packets for a single video.

    Args:
        video_id (str): Video identifier.
        video_h5 (h5py.File): HDF5 file with video features.

    Returns:
        Tuple[int, int]: Counts of audio and video packets created.
    """
    input_file = os.path.join(INPUT_PATH, f"{video_id}.avi")
    tmp_video_dir = os.path.join('data/_tmp_video', video_id)
    tmp_audio_wav = os.path.join('data/_tmp_audio', f"{video_id}.wav")

    ensure_dirs([tmp_video_dir, 'data/_tmp_audio'])

    extract_audio_ffmpeg(input_file, tmp_audio_wav)
    extract_video_frames_ffmpeg(input_file, tmp_video_dir)

    audio_packets = load_audio_packets(tmp_audio_wav)
    audio_packet_objs = save_audio_packets(audio_packets, OUTPUT_PATH, video_id)
    video_packet_objs = save_video_packets(tmp_video_dir, OUTPUT_PATH, video_id, video_h5=video_h5)

    cleanup_temp_files(tmp_dirs=[tmp_video_dir], tmp_files=[tmp_audio_wav])

    return len(audio_packet_objs), len(video_packet_objs)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    annotations = pd.read_csv(ANNOTATIONS_PATH)
    test_video_ids = annotations[annotations['split'] == 'test']['video_id'].unique().tolist()

    with h5py.File(VIDEO_FEATURES_PATH, 'r') as video_h5:
        all_audio_packet_counts = []
        all_video_packet_counts = []
        for vid in tqdm(test_video_ids, desc="Processing test videos", ascii=True):
            try:
                a_count, v_count = process_single_video(vid, video_h5)
                all_audio_packet_counts.append(a_count)
                all_video_packet_counts.append(v_count)
            except Exception as e:
                logger.error(f"Error processing {vid}: {e}")

    logger.info("\nSummary of packet counts:")
    for i, vid in enumerate(test_video_ids):
        logger.info(f"{vid}: Audio packets={all_audio_packet_counts[i]}, Video packets={all_video_packet_counts[i]}")

    logger.info("\nAverage packet counts:")
    logger.info(f"Average Audio Packets: {np.mean(all_audio_packet_counts):.2f}")
    logger.info(f"Average Video Packets: {np.mean(all_video_packet_counts):.2f}")


if __name__ == '__main__':
    main()