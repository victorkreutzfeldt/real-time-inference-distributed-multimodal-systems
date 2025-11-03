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

from src.vggish_input import waveform_to_examples
from src._class_pipeline_audio import AudioPipeline
from src.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# ====================== Config ======================
ANNOTATIONS_PATH = 'data/annotations.csv'
INPUT_PATH = 'data/AVE_trimmed'

AUDIO_SAMPLING_RATE = 16000 # audio sampling rate
NUM_TOKENS = 10 # number of tokens to split audio into
TOKEN_DURATION = 1.0  # 1 second per token
NUM_EXPECTED_AUDIO_SAMPLES = int(AUDIO_SAMPLING_RATE * TOKEN_DURATION * NUM_TOKENS) # expected number of samples for the entire video

# Define paths for output
OUTPUT_WAVEFORMS_PATH = 'data/raw/audio_waveforms.h5'

OUTPUT_SPECTROGRAMS_COR96_PATH = 'data/raw/spectrograms/audio_cor96.h5'
OUTPUT_SPECTROGRAMS_UNC40_PATH = 'data/raw/spectrograms/audio_unc40.h5' 
OUTPUT_SPECTROGRAMS_UNC16_PATH = 'data/raw/spectrograms/audio_unc16.h5' 

OUTPUT_FEATURES_PATH = 'data/classification/features/audio.h5'
OUTPUT_FEATURES_SPA_GPA_PATH = 'data/classification/features/audio_spa_gpa.h5'

OUTPUT_EMBEDDINGS_PATH = 'data/classification/embeddings/audio.h5'
OUTPUT_EMBEDDINGS_PCA_PATH = 'data/classification/embeddings/audio_pca.h5'

OUTPUT_TIMING_PATH = 'data/classification/timing/audio_pipeline.pth'

# Check if output directories exist, if not create them
os.makedirs(os.path.dirname(OUTPUT_WAVEFORMS_PATH), exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_SPECTROGRAMS_COR96_PATH), exist_ok=True)
#os.makedirs(os.path.dirname(OUTPUT_SPECTROGRAMS_UNC40_PATH), exist_ok=True)
#os.makedirs(os.path.dirname(OUTPUT_SPECTROGRAMS_UNC16_PATH), exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)

os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_PATH), exist_ok=True)
#os.makedirs(os.path.dirname(OUTPUT_EMBEDDINGS_PCA_PATH), exist_ok=True)

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

# ====================== Functions ======================
def extract_audio_waveform(file_path, target_num_samples=NUM_EXPECTED_AUDIO_SAMPLES):
    """Extract audio waveform from a video file and ensure it has the expected number of samples."""

    # Open video file
    container = av.open(file_path)
    
    # Get audio stream
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        RuntimeError(f"[ERROR] No audio stream found in {file_path}.")

    # Read audio samples
    waveform = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            samples = frame.to_ndarray().reshape(-1, frame.layout.nb_channels).astype(np.float32)
            # Convert to mono if multichannel
            if frame.layout.nb_channels > 1:
                samples = np.mean(samples, axis=1)
            waveform.append(samples)
    container.close()

    # Concatenate all samples
    waveform = np.concatenate(waveform, axis=0).squeeze()
    #breakpoint()
    # Normalize waveform to [-1, 1]
    waveform = waveform / 32768 
    
    # Padding if necessary
    if len(waveform) > target_num_samples:
        # Truncates if it is longer than expected
        waveform = waveform[:target_num_samples]    
    elif len(waveform) < target_num_samples:
        # Pad zeros at the end
        pad_width = target_num_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant', constant_values=0)
    
    if len(waveform) == 0:
        RuntimeError(f"[ERROR] No valid audio waveform extracted from {file_path}.")
    
    return waveform

# ====================== Main ======================
if __name__ == "__main__":
    
    # Load VGGish feature extractor model
    audio_pipeline = AudioPipeline(pretrained=True, device=DEVICE, preprocess=False, postprocess=True)
    audio_pipeline.eval()
    
    # Get video files
    video_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('.avi')]
    
    # Prepare to save timing information
    timing = {'waveform': [], 'spectogram': [], 'processing': []}    

    # ====================== Feature Extraction ======================

    # Open HDF5 file once and save per video
    with h5py.File(OUTPUT_WAVEFORMS_PATH, 'w') as hf_waveforms,\
        h5py.File(OUTPUT_SPECTROGRAMS_COR96_PATH, 'w') as hf_spectrogram_cor96,\
        h5py.File(OUTPUT_SPECTROGRAMS_UNC40_PATH, 'w') as hf_spectrogram_unc40,\
        h5py.File(OUTPUT_SPECTROGRAMS_UNC16_PATH, 'w') as hf_spectrogram_unc16,\
        h5py.File(OUTPUT_FEATURES_PATH, 'w') as hf_feats,\
        h5py.File(OUTPUT_FEATURES_SPA_GPA_PATH, 'w') as hf_feats_spa_gpa,\
        h5py.File(OUTPUT_EMBEDDINGS_PATH, 'w') as hf_embs,\
        h5py.File(OUTPUT_EMBEDDINGS_PCA_PATH, 'w') as hf_embs_pca:

        # Iterate over video files
        for idx, name in enumerate(tqdm(video_files, ascii=True)):
            
            # Get file path
            file_path = os.path.join(INPUT_PATH, name)

            # Get video ID from filename
            video_id = os.path.splitext(name)[0]

            # Extract audio waveform
            time_start = time.time()
            waveforms = extract_audio_waveform(file_path, target_num_samples=NUM_EXPECTED_AUDIO_SAMPLES)
            time_end = time.time()

            # Save timing for extraction
            timing['waveform'].append((time_end - time_start) / NUM_TOKENS)

            # Resize waveform to fit into ten chunks
            waveforms = waveforms.reshape(NUM_TOKENS, -1)
        
            # Prepare to collect data
            spectrograms_cor96 = []
        
            # Prepare to save times for extracting the waveform
            timing_example = []

            # Iterate over chunks
            for cc in range(NUM_TOKENS):
                
                # Get chunk waveform 
                chunk_waveform = waveforms[cc]
            
                # Start timing for waveform to spectrogram conversion
                time_start = time.time()

                # Convert waveform to spectrogram
                spectrogram_cor96 = waveform_to_examples(data=chunk_waveform, sample_rate=AUDIO_SAMPLING_RATE)
            
                # End timing for waveform to spectrogram conversion
                time_end = time.time()

                # Store timing for this chunk
                timing_example.append(time_end - time_start)

                # Append data
                spectrograms_cor96.append(spectrogram_cor96)

            # Store average timing for spectrogram extraction
            timing['spectogram'].append(float(np.mean(timing_example)))                
    
            # Extract uncorrelated spectrograms
            spectrograms_unc40 = []
            spectrograms_unc16 = []

            for cc in range(NUM_TOKENS):
                
                # Get chunk waveform 
                chunk_waveform = waveforms[cc]
            
                # Start timing for waveform to spectrogram conversion
                time_start = time.time()

                # Convert waveform to spectrogram
                spectogram_unc40 = waveform_to_examples(data=chunk_waveform, sample_rate=AUDIO_SAMPLING_RATE, 
                                               stft_window_length_seconds=0.025, stft_hop_length_seconds=0.025,
                                               example_window_seconds=1.0, example_hop_seconds=1.0)
                spectogram_unc16 = waveform_to_examples(
                                                data=chunk_waveform, sample_rate=AUDIO_SAMPLING_RATE, 
                                                stft_window_length_seconds=0.0625, stft_hop_length_seconds=0.0625,
                                                example_window_seconds=1.0, example_hop_seconds=1.0)
            
                # End timing for waveform to spectrogram conversion
                time_end = time.time()

                # Store timing for this chunk
                timing_example.append(time_end - time_start)

                # Append data
                spectrograms_unc40.append(spectogram_unc40)
                spectrograms_unc16.append(spectogram_unc16) 

            # Stack data into a single numpy array
            spectrograms_cor96 = np.stack(spectrograms_cor96)
            spectrograms_unc40 = np.stack(spectrograms_unc40)
            spectrograms_unc16 = np.stack(spectrograms_unc16)

            # Convert spectrograms to torch tensor
            spectrograms_cor96_tensor = torch.tensor(spectrograms_cor96, dtype=torch.float32).to(DEVICE)

            # Extract features using VGGish
            with torch.no_grad():
                time_start = time.time()
                out = audio_pipeline(spectrograms_cor96_tensor, return_feats=True, return_embs=True)
                time_end = time.time()
                timing['processing'].append((time_end - time_start) / NUM_TOKENS)
                
            # Take off channel dimension of spectrogram
            spectrograms_cor96 = spectrograms_cor96.squeeze(1)
            spectrograms_unc40 = spectrograms_unc40.squeeze(1)
            spectrograms_unc16 = spectrograms_unc16.squeeze(1)

            # Get features and embeddings
            feats = out['feats']
            embs = out['embs']
            embs_pca = out['embs_pca']

            # Convert to numpy array
            feats = feats.cpu().numpy().astype(np.float32) 
            embs = embs.cpu().numpy().astype(np.float32)
            embs_pca = embs_pca.cpu().numpy().astype(np.uint8)
        
            # Compute different spatial GPA
            feats_spa_gpa = feats.mean(axis=(-1, -2))

            # Save to output files
            #hf_waveforms.create_dataset(video_id, data=waveforms, dtype=np.float32, compression="gzip")

            hf_spectrogram_cor96.create_dataset(video_id, data=spectrograms_cor96, dtype=np.float32, compression="gzip")
            hf_spectrogram_unc40.create_dataset(video_id, data=spectrograms_unc40, dtype=np.float32, compression="gzip")
            hf_spectrogram_unc16.create_dataset(video_id, data=spectrograms_unc16, dtype=np.float32, compression="gzip")

            hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
            hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")

            hf_embs.create_dataset(video_id, data=embs, dtype=np.float32, compression="gzip")
            hf_embs_pca.create_dataset(video_id, data=embs_pca, dtype=np.uint8, compression="gzip")

    # Save timing information
    with open(OUTPUT_TIMING_PATH, 'wb') as f:
        pickle.dump(timing, f)
