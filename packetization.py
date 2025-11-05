#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

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


# ---- Config ----
TARGET_FPS = 16
TARGET_SAMPLING_RATE = 16000
TARGET_VIDEO_DURATION = 10.0
IMAGE_RESOLUTION = (224, 224)

TARGET_NUM_VIDEO_PACKETS = int(TARGET_FPS * TARGET_VIDEO_DURATION)  # 160 visual samples
AUDIO_WINDOW_SIZE = 320 # samples per audio packet
TARGET_NUM_AUDIO_PACKETS = int(TARGET_SAMPLING_RATE * TARGET_VIDEO_DURATION / AUDIO_WINDOW_SIZE)

ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'
VIDEO_FEATURES_PATH = 'data/classification/features/video.h5'

OUTPUT_PATH = 'data/packets'


# ---- Utility Functions ----
def ensure_dirs(dirs: List[str]) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def extract_video_frames_ffmpeg(input_path: str, tmp_dir: str) -> None:
    """
    Extract video frames at TARGET_FPS and resize using FFmpeg to tmp_dir as PNG files.

    Args:   
        input_path (str): Path to input video file.
        tmp_dir (str): Directory to save extracted frames.  

    Returns:
        None    
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Remove any old frames first
    for f in os.listdir(tmp_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(tmp_dir, f))

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'quiet',
        '-i', input_path,
        '-vf', f'fps={TARGET_FPS},scale={IMAGE_RESOLUTION[0]}:{IMAGE_RESOLUTION[1]}',
        os.path.join(tmp_dir, 'frame_%05d.png')
    ]
    subprocess.run(cmd, check=True)


def extract_audio_ffmpeg(input_path: str, tmp_wav_path: str) -> None:
    """
    Extract mono audio downsampled to TARGET_SAMPLING_RATE as WAV file.

    Args:
        input_path (str): Path to input video file.     
        tmp_wav_path (str): Path to output WAV file.

    Returns:
        None
    """
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'quiet',
        '-i', input_path,
        '-ac', '1',  # mono
        '-ar', str(TARGET_SAMPLING_RATE),
        '-vn',       # no video
        tmp_wav_path
    ]
    subprocess.run(cmd, check=True)


def load_audio_packets(wav_path: str, window_size: int = AUDIO_WINDOW_SIZE, target_packets: int = TARGET_NUM_AUDIO_PACKETS) -> Tuple[np.ndarray, int]:
    """
    Load audio from WAV file and split into packets of fixed window size.   

    Args:
        wav_path (str): Path to input WAV file. 
        window_size (int): Number of samples per audio packet.
        target_packets (int): Number of audio packets to extract.

    Returns:
        np.ndarray: Array of shape (target_packets, window_size) containing audio samples.
        int: Sampling rate of the audio.
    """
    # Get data and sampling rate
    sr, data = scipy.io.wavfile.read(wav_path)
    
    # Normalize int16 audio to float32 [-1,1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    # Pad or truncate to expected length
    total_samples = window_size * target_packets
    
    # Sanity checks
    assert sr == TARGET_SAMPLING_RATE, f"Expected {TARGET_SAMPLING_RATE}, got {sr}."
    assert data.ndim == 1, f"Expected mono audio, got {data.ndim} channels."

    if len(data) < total_samples:
        data = np.pad(data, (0, total_samples - len(data)), mode='constant')
    else:
        data = data[:total_samples]

    # Get packets
    packets = data.reshape((target_packets, window_size))
    
    return packets, sr


def save_audio_packets(audio_packets: np.ndarray, save_dir: str, video_id: str) -> List[Packet]:
    """
    Save audio packets to gzipped pickle file, where each audio packet is a Packet object with metadata. Each packet contains AUDIO_WINDOW_SIZE samples
    encoded using PCM-16.

    Args:
        audio_packets (np.ndarray): Array of shape (num_packets, AUDIO_WINDOW_SIZE) containing audio samples.
        save_dir (str): Directory to save the packets.      
        video_id (str): Identifier for the video/audio.
    
    Returns:
        List[Packet]: List of Packet objects representing the audio packets.
    """
    # Define path
    packets_path = os.path.join(save_dir, f'audio/{video_id}.pkl.gz')
    
    # Prepare packet list
    packets = []
    
    # Compute packet duration as Fraction
    packet_duration = Fraction(AUDIO_WINDOW_SIZE, TARGET_SAMPLING_RATE)

    # Compute packet size in bits
    size_bits = AUDIO_WINDOW_SIZE * 16 # int16 samples, 16 bits each

    # Iterate over packets
    for cc, pp in enumerate(audio_packets):

        # Sort out metadata
        pts = cc
        pts_time = packet_duration * cc  
        duration = packet_duration     

        packet = Packet(
            stream_type='audio',
            pts=pts,
            pts_time=pts_time,       # Fraction type stored here
            duration=duration,       # Fraction as well
            size_bits=size_bits,
            sample_rate=TARGET_SAMPLING_RATE,
            time_base=Fraction(1, TARGET_SAMPLING_RATE),
            nb_channels=1,
            payload=pp.astype(np.float32)
        )
        packets.append(packet)

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)

    return packets


def save_video_packets(frame_dir: str, save_dir: str, video_id: str, video_h5: Optional[h5py.File] = None) -> List[Packet]:
    """
    Save video packets to gzipped pickle file, where each video packet is a Packet object with metadata. Each packet corresponds to a video frame encoded 
    using PNG. To save computational resources, we already use the visual features stored in video_h5 as payloads instead of raw images.

    Args:
        frame_dir (str): Directory containing extracted video frames as PNG files.
        save_dir (str): Directory to save the packets.      
        video_id (str): Identifier for the video.
        video_h5 (h5py.File, optional): H5 file containing video features to load payloads from. If None, payloads are not loaded.
    
    Returns:
        List[Packet]: List of Packet objects representing the video packets.
    """
    # Define path
    packets_path = os.path.join(save_dir, f'video/{video_id}.pkl.gz')

    # Read and sort frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    # Ensure we have the expected number of frames
    if len(frame_files) < TARGET_NUM_VIDEO_PACKETS:
        raise RuntimeError(f"Not enough frames extracted: got {len(frame_files)}, expected {TARGET_NUM_VIDEO_PACKETS}")
    elif len(frame_files) > TARGET_NUM_VIDEO_PACKETS:
        frame_files = frame_files[:TARGET_NUM_VIDEO_PACKETS]

    # Get payloads from H5 if provided
    if video_h5 is not None:
        video_payloads = np.array(video_h5[video_id][:]).reshape(TARGET_NUM_VIDEO_PACKETS, 512, 7, 7)
    else:
        video_payloads = None

    # Load frames and create packets
    packets = []

    # Compute packet duration as Fraction
    packet_duration = Fraction(1, TARGET_FPS) 

    # Iterate over frames
    for ff, fname in enumerate(frame_files):
        
        # Get frame path
        path = os.path.join(frame_dir, fname)

        # Load image
        img = Image.open(path).convert('RGB')
        np_img = np.array(img, dtype=np.uint8)
        
        pts = ff
        pts_time = packet_duration * ff
        size_bits = np_img.nbytes * 8

        packet = Packet(
            stream_type='video',
            pts=pts,
            pts_time=pts_time,
            duration=packet_duration,
            size_bits=size_bits,
            sample_rate=TARGET_FPS,
            time_base=Fraction(1, TARGET_FPS),
            nb_channels=3,
            resolution=IMAGE_RESOLUTION[::-1],
            payload=video_payloads[ff] if video_payloads is not None else np_img
        )
        packets.append(packet)

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)

    return packets


def cleanup_temp_files(tmp_dirs: List[str], tmp_files: List[str]) -> None:
    """
    Remove temporary directories and files.
    """
    # Remove temporary files
    for f in tmp_files:
        if os.path.isfile(f):
            os.remove(f)
    # Remove temporary directories
    for d in tmp_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def process_single_video(video_id: str, video_h5: h5py.File) -> Tuple[int, int]:
    """
    Process a single video: extract audio and video, save packets, cleanup temporary data.
    
    Args:
        video_id (str): ID of the video to process.
        video_h5 (h5py.File): Opened HDF5 file for video features.
    
    Returns:
        Tuple[int, int]: Number of audio packets, number of video packets.
    """
    input_file = os.path.join(INPUT_PATH, f"{video_id}.avi")
    tmp_video_dir = os.path.join('data/_tmp_video', video_id)
    tmp_audio_wav = os.path.join('data/_tmp_audio', f"{video_id}.wav")

    ensure_dirs([tmp_video_dir, 'data/_tmp_audio'])

    extract_audio_ffmpeg(input_file, tmp_audio_wav)
    extract_video_frames_ffmpeg(input_file, tmp_video_dir)

    audio_packets, sr = load_audio_packets(tmp_audio_wav)
    audio_pkt_objs = save_audio_packets(audio_packets, OUTPUT_PATH, video_id)
    video_pkt_objs = save_video_packets(tmp_video_dir, OUTPUT_PATH, video_id, video_h5=video_h5)

    cleanup_temp_files(tmp_dirs=[tmp_video_dir], tmp_files=[tmp_audio_wav])

    return len(audio_pkt_objs), len(video_pkt_objs)

# ---- Main ----
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