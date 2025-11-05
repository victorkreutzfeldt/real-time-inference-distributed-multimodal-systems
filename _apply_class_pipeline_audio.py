#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import pickle
import time
import logging

import numpy as np
import torch
import av
import h5py
from tqdm import tqdm

from src.vggish_input import waveform_to_examples
from src._class_pipeline_audio import AudioPipeline
from src.utils import set_seed


# Set random seed for reproducibility
set_seed(42)

# ---- Config ----
ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'

TARGET_SAMPLING_RATE = 16000  # audio sampling rate
NUM_TOKENS = 10  # number of tokens to split audio into
TOKEN_DURATION = 1.0  # 1 second per token
TARGET_NUM_AUDIO_SAMPLES = int(TARGET_SAMPLING_RATE * TOKEN_DURATION * NUM_TOKENS)  # expected number of samples for entire video

OUTPUT_WAVEFORMS_PATH = 'data/raw/audio_waveforms.h5'
OUTPUT_SPECTROGRAMS_COR96_PATH = 'data/raw/spectrograms/audio_cor96.h5'
OUTPUT_FEATURES_PATH = 'data/classification/features/audio.h5'
OUTPUT_FEATURES_SPA_GPA_PATH = 'data/classification/features/audio_spa_gpa.h5'
OUTPUT_EMBEDDINGS_PATH = 'data/classification/embeddings/audio.h5'
OUTPUT_EMBEDDINGS_PCA_PATH = 'data/classification/embeddings/audio_pca.h5'
OUTPUT_TIMING_PATH = 'data/classification/timing/audio_pipeline.pth'

# Ensure output directories exist
output_dirs = set(os.path.dirname(p) for p in [
    OUTPUT_WAVEFORMS_PATH, OUTPUT_SPECTROGRAMS_COR96_PATH, OUTPUT_FEATURES_PATH,
    OUTPUT_FEATURES_SPA_GPA_PATH, OUTPUT_EMBEDDINGS_PATH, OUTPUT_EMBEDDINGS_PCA_PATH,
    OUTPUT_TIMING_PATH
])
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# Determine computation device
DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    else torch.device('cpu')
)

NUM_CPUS = 2 if DEVICE.type == 'cuda' else 0
PIN_MEMORY = DEVICE.type == 'cuda'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"[INFO] Using device: {DEVICE}")


def extract_audio_waveform(file_path: str, target_num_samples: int = TARGET_NUM_AUDIO_SAMPLES) -> np.ndarray:
    """
    Extract audio waveform from a video file and ensure it has the expected number of samples.

    Args: 
        file_path (str): Path to input video file.
        target_num_samples (int): Expected number of audio samples in output.

    Returns:
        np.ndarray: Waveform array of length target_num_samples.
    """
    container = av.open(file_path)
    
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        raise RuntimeError(f"[ERROR] No audio stream found in {file_path}.")

    waveform = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            samples = frame.to_ndarray().reshape(-1, frame.layout.nb_channels).astype(np.float32)
            if frame.layout.nb_channels > 1:
                samples = np.mean(samples, axis=1)
            waveform.append(samples)
    container.close()

    waveform = np.concatenate(waveform, axis=0).squeeze()
    waveform = waveform / 32768  # Normalize

    if len(waveform) > target_num_samples:
        waveform = waveform[:target_num_samples]
    elif len(waveform) < target_num_samples:
        pad_width = target_num_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant', constant_values=0)

    if len(waveform) == 0:
        raise RuntimeError(f"[ERROR] No valid audio waveform extracted from {file_path}.")

    return waveform


def main():
    video_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.avi')]

    timing = {'waveform': [], 'spectrogram': [], 'processing': []}

    with h5py.File(OUTPUT_WAVEFORMS_PATH, 'w') as hf_waveforms, \
         h5py.File(OUTPUT_SPECTROGRAMS_COR96_PATH, 'w') as hf_spectrogram_cor96, \
         h5py.File(OUTPUT_FEATURES_PATH, 'w') as hf_feats, \
         h5py.File(OUTPUT_FEATURES_SPA_GPA_PATH, 'w') as hf_feats_spa_gpa, \
         h5py.File(OUTPUT_EMBEDDINGS_PATH, 'w') as hf_embs, \
         h5py.File(OUTPUT_EMBEDDINGS_PCA_PATH, 'w') as hf_embs_pca:

        audio_pipeline = AudioPipeline(pretrained=True, device=DEVICE, preprocess=False, postprocess=True)
        audio_pipeline.eval()

        for idx, name in enumerate(tqdm(video_files, ascii=True)):
            file_path = os.path.join(INPUT_PATH, name)
            video_id = os.path.splitext(name)[0]

            # Extract waveform with timing
            start_t = time.time()
            waveforms = extract_audio_waveform(file_path, target_num_samples=TARGET_NUM_AUDIO_SAMPLES)
            end_t = time.time()
            timing['waveform'].append((end_t - start_t) / NUM_TOKENS)

            # Reshape to NUM_TOKENS x samples_per_token
            waveforms = waveforms.reshape(NUM_TOKENS, -1)

            spectrograms_cor96 = []
            timing_example = []

            # Convert waveforms to spectrograms per token chunk
            for tt in range(NUM_TOKENS):
                waveform_token = waveforms[tt]
                t_start = time.time()
                spectrogram_cor96 = waveform_to_examples(waveform_token, sample_rate=TARGET_SAMPLING_RATE)
                t_end = time.time()
                timing_example.append(t_end - t_start)
                spectrograms_cor96.append(spectrogram_cor96)

            timing['spectrogram'].append(float(np.mean(timing_example)))

            spectrograms_cor96 = np.stack(spectrograms_cor96)
            spectrograms_cor96_tensor = torch.tensor(spectrograms_cor96, dtype=torch.float32).to(DEVICE)

            # Feature extraction
            with torch.no_grad():
                t_start = time.time()
                out = audio_pipeline(spectrograms_cor96_tensor, return_feats=True, return_embs=True)
                t_end = time.time()
                timing['processing'].append((t_end - t_start) / NUM_TOKENS)

            spectrograms_cor96 = spectrograms_cor96.squeeze(1)

            feats = out['feats'].cpu().numpy().astype(np.float32)
            embs = out['embs'].cpu().numpy().astype(np.float32)
            embs_pca = out['embs_pca'].cpu().numpy().astype(np.uint8)
            feats_spa_gpa = feats.mean(axis=(-1, -2))

            # Save datasets to HDF5

            # Uncomment if saving raw audio needed
            # hf_waveforms.create_dataset(video_id, data=waveforms, dtype=np.float32, compression="gzip")

            hf_spectrogram_cor96.create_dataset(video_id, data=spectrograms_cor96, dtype=np.float32, compression="gzip")
            hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
            hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")
            hf_embs.create_dataset(video_id, data=embs, dtype=np.float32, compression="gzip")
            hf_embs_pca.create_dataset(video_id, data=embs_pca, dtype=np.uint8, compression="gzip")

    # Save timing information as pickle
    with open(OUTPUT_TIMING_PATH, 'wb') as f:
        pickle.dump(timing, f)


if __name__ == "__main__":
    main()
