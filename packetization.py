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

from fractions import Fraction
from src.packets import Packet

# Configuration
TARGET_FPS = 16
TARGET_SAMPLING_RATE = 16000
TARGET_VIDEO_DURATION = 10.0
IMAGE_RESOLUTION = (224, 224)

TARGET_NUM_VIDEO_PACKETS = int(TARGET_FPS * TARGET_VIDEO_DURATION)  # 160 frames

AUDIO_WINDOW_SIZE = 320 # samples per audio packet
TARGET_NUM_AUDIO_PACKETS = int(TARGET_SAMPLING_RATE * TARGET_VIDEO_DURATION / AUDIO_WINDOW_SIZE)

ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'
VIDEO_FEATURES_PATH = 'data/classification/features/video.h5'

OUTPUT_PATH = 'data/packets'
os.makedirs(os.path.join(OUTPUT_PATH, 'audio'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'video'), exist_ok=True)

def extract_video_frames_ffmpeg(input_path, tmp_dir):
    """Extract video frames at TARGET_FPS and resize using FFmpeg to tmp_dir as PNG files."""
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


def extract_audio_ffmpeg(input_path, tmp_wav_path):
    """Extract mono audio downsampled to TARGET_SAMPLING_RATE as WAV file."""

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'quiet',
        '-i', input_path,
        '-ac', '1',  # mono
        '-ar', str(TARGET_SAMPLING_RATE),
        '-vn',       # no video
        tmp_wav_path
    ]
    subprocess.run(cmd, check=True)


def load_audio_chunks(wav_path, window_size=AUDIO_WINDOW_SIZE, target_packets=TARGET_NUM_AUDIO_PACKETS):
    
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

    # Chunk into windows
    chunks = data.reshape((target_packets, window_size))
    
    return chunks, sr


def save_audio_packets(audio_chunks, save_dir, video_id):
    
    # Define paths
    payload_path = os.path.join(save_dir, f'audio/{video_id}_payloads.h5')
    packets_path = os.path.join(save_dir, f'audio/{video_id}_packets.pkl.gz')
    pts2idx_path = os.path.join(save_dir, f'audio/{video_id}_pts2idx.pkl.gz')
    
    # Prepare payloads and packets
    packets = []
    pts2idx = {}
    
    # Compute packet duration as Fraction
    packet_duration = Fraction(AUDIO_WINDOW_SIZE, TARGET_SAMPLING_RATE)

    # Compute packet size in bits
    size_bits = AUDIO_WINDOW_SIZE * 16 # int16 samples, 16 bits each

    # Iterate over chunks and create packets
    for cc, chunk in enumerate(audio_chunks):

        # Sort out metadata
        pts = cc
        pts_time = packet_duration * cc  # Fraction multiplication, exact rational time
        duration = packet_duration       # Store duration also as Fraction if desired

        packet = Packet(
            stream_type='audio',
            pts=pts,
            pts_time=pts_time,       # Fraction type stored here
            duration=duration,       # Fraction as well
            size_bits=size_bits,
            sample_rate=TARGET_SAMPLING_RATE,
            time_base=Fraction(1, TARGET_SAMPLING_RATE),
            nb_channels=1
        )

        packets.append(packet)
        pts2idx[pts] = cc
    
    with h5py.File(payload_path, 'w') as f:
        dt = h5py.vlen_dtype(np.float32)
        dset = f.create_dataset('payloads', (len(audio_chunks),), dtype=dt)
        dset[:] = [chunk for chunk in audio_chunks]

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)
    
    with gzip.open(pts2idx_path, 'wb') as f:
        pickle.dump(pts2idx, f)

    return packets


def save_video_packets(frame_dir, save_dir, video_id, video_h5=None):

    # Define paths
    payload_path = os.path.join(save_dir, f'video/{video_id}_payloads.h5')
    packets_path = os.path.join(save_dir, f'video/{video_id}_packets.pkl.gz')
    pts2idx_path = os.path.join(save_dir, f'video/{video_id}_pts2idx.pkl.gz')

    # Read and sort frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    # Ensure we have the expected number of frames
    if len(frame_files) < TARGET_NUM_VIDEO_PACKETS:
        raise RuntimeError(f"Not enough frames extracted: got {len(frame_files)}, expected {TARGET_NUM_VIDEO_PACKETS}")
    elif len(frame_files) > TARGET_NUM_VIDEO_PACKETS:
        frame_files = frame_files[:TARGET_NUM_VIDEO_PACKETS]

    # Load frames and create packets
    video_payloads = []
    packets = []
    pts2idx = {}

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
            resolution=IMAGE_RESOLUTION[::-1]
        )
        packets.append(packet)
        pts2idx[pts] = ff

    if not None:
        video_payloads = video_h5[video_id][:]
    video_payloads = video_payloads.reshape((TARGET_NUM_VIDEO_PACKETS, *video_payloads.shape[2:]))

    with h5py.File(payload_path, 'w') as f:
        f.create_dataset('payloads', data=video_payloads, compression='gzip')

    with gzip.open(packets_path, 'wb') as f:
        pickle.dump(packets, f)
    
    with gzip.open(pts2idx_path, 'wb') as f:
        pickle.dump(pts2idx, f)
    
    return packets


if __name__ == '__main__':
    
    # Read annotations to get test video IDs
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    test_video_ids = annotations[annotations['split'] == 'test']['video_id'].unique().tolist()

    # Open video features file
    video_h5 = h5py.File(VIDEO_FEATURES_PATH, 'r')

    # For recording packet counts
    all_audio_packet_counts = []
    all_video_packet_counts = []

    # Interate over test videos
    for video_id in tqdm(test_video_ids, desc="Processing test videos", ascii=True):
        
        # Get input file path
        input_file = os.path.join(INPUT_PATH, f"{video_id}.avi")
        
        # Temporary directories and files
        tmp_frame_dir = os.path.join('data/_tmp_frames', video_id)
        tmp_audio_wav = os.path.join('data/_tmp_audio', f"{video_id}.wav")
        
        os.makedirs(tmp_frame_dir, exist_ok=True)
        os.makedirs('data/_tmp_audio', exist_ok=True)
        
        # Extract audio wav
        extract_audio_ffmpeg(input_file, tmp_audio_wav)

        # Extract video frames
        extract_video_frames_ffmpeg(input_file, tmp_frame_dir)
        
        # Load and chunk audio
        audio_chunks, sr = load_audio_chunks(tmp_audio_wav)

        # Save packets
        audio_packets = save_audio_packets(audio_chunks, OUTPUT_PATH, video_id)
        video_packets = save_video_packets(tmp_frame_dir, OUTPUT_PATH, video_id, video_h5=video_h5)
     
        # Record counts
        all_audio_packet_counts.append(len(audio_packets))
        all_video_packet_counts.append(len(video_packets))

        # Cleanup temporary files/folders
        try:
            if os.path.isfile(tmp_audio_wav):
                os.remove(tmp_audio_wav)

            if os.path.isdir(tmp_frame_dir):
                shutil.rmtree(tmp_frame_dir)
                
        except Exception as e:
            print(f"Warning cleaning temp files for {video_id}: {e}")

    print("\nSummary of packet counts:")
    for i, vid in enumerate(test_video_ids):
        print(f"{vid}: Audio packets={all_audio_packet_counts[i]}, Video packets={all_video_packet_counts[i]}")

    print("\nAverage packet counts:")
    print(f"Average Audio Packets: {np.mean(all_audio_packet_counts):.2f}")
    print(f"Average Video Packets: {np.mean(all_video_packet_counts):.2f}")
