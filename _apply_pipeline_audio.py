#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio feature extraction pipeline script.

This script extracts audio waveforms from video files, splits them into fixed-length
tokens, converts tokens into spectrograms, extracts deep audio features and embeddings
using a pretrained AudioPipeline wrapped for modular processing, and saves all intermediate
data to HDF5 files for later training and analysis.

It also logs timing information for waveform extraction, spectrogram computation,
and feature extraction to facilitate performance profiling.

Constants for file paths, sampling rates, and processing parameters are defined in a config
dictionary inside the main() function.

Main Components:
    - AudioPipelineWrapper: Encapsulates waveform loading, tokenization, spectrogram extraction,
  and embedding computation using a pretrained AudioPipeline.
    - main: Drives batch processing over a directory of videos.

Outputs:
    - HDF5 files containing processed data saved at configured paths:
        * Raw audio waveforms (NOTE: commented out because consume too much disk space).
        * Spectrograms for each token.
        * Extracted features and spatially pooled features.
        * Audio embeddings and PCA-processed embeddings.
    - Pickle file recording processing timing information per video.

Usage:
    Simply run:
    python _apply_pipeline_audio.py

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import os
import pickle
import time

import numpy as np
import torch
import av
import h5py
from tqdm import tqdm

from src.vggish_input import waveform_to_examples

from src.pipeline_audio import AudioPipeline

from src.utils import set_seed


class AudioPipelineWrapper:
    """
    Wrapper for audio feature extraction pipeline supporting waveform extraction,
    tokenization, spectrogram and embedding extraction, and batch processing over a dataset.

    Args:
        device (torch.device): Device for computation.
        pretrained (bool): Whether to load pretrained weights.
        preprocess (bool): Whether to preprocess raw inputs.
        postprocess (bool): Whether to apply PCA whitening and quantization postprocessing.
    """

    def __init__(self, device: torch.device, pretrained: bool = True, preprocess: bool = False, postprocess: bool = True):
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.pipeline = AudioPipeline(pretrained=pretrained, device=device, preprocess=preprocess, postprocess=postprocess)
        self.pipeline.eval()

    def extract_audio_waveform(self, file_path: str, target_num_samples: int) -> np.ndarray:
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

    def process_tokens(self, waveforms: np.ndarray, target_sampling_rate: int, num_tokens: int):
        waveforms = waveforms.reshape(num_tokens, -1)

        spectrograms = []
        timing_example = []

        for tt in range(num_tokens):
            waveform_token = waveforms[tt]
            t_start = time.time()
            spectrogram = waveform_to_examples(waveform_token, sampling_rate=target_sampling_rate)
            t_end = time.time()
            timing_example.append(t_end - t_start)
            spectrograms.append(spectrogram)

        spectrograms = np.stack(spectrograms).astype(np.float32)
        spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            t_start = time.time()
            out = self.pipeline(spectrograms_tensor, return_feats=True, return_embs=True)
            t_end = time.time()
            out['timing'] = (t_end - t_start) / num_tokens

        out['spectrograms'] = spectrograms

        return out, timing_example

    def process_videos(self,
                       input_path: str,
                       num_tokens: int,
                       token_duration: float,
                       target_sampling_rate: int,
                       output_waveforms_path: str,
                       output_spectrograms_path: str,
                       output_features_path: str,
                       output_features_spa_gpa_path: str,
                       output_embeddings_path: str,
                       output_embeddings_pca_path: str,
                       output_timing_path: str):

        expected_num_samples = int(target_sampling_rate * token_duration * num_tokens)
        video_files = [f for f in os.listdir(input_path) if f.endswith('.avi')]

        timing = {'waveform': [], 'spectrogram': [], 'processing': []}

        with h5py.File(output_waveforms_path, 'w') as hf_waveforms, \
             h5py.File(output_spectrograms_path, 'w') as hf_spectrograms, \
             h5py.File(output_features_path, 'w') as hf_feats, \
             h5py.File(output_features_spa_gpa_path, 'w') as hf_feats_spa_gpa, \
             h5py.File(output_embeddings_path, 'w') as hf_embs, \
             h5py.File(output_embeddings_pca_path, 'w') as hf_embs_pca:

            for idx, name in enumerate(tqdm(video_files, ascii=True)):
                file_path = os.path.join(input_path, name)
                video_id = os.path.splitext(name)[0]

                start_t = time.time()
                waveforms = self.extract_audio_waveform(file_path, target_num_samples=expected_num_samples)
                end_t = time.time()
                timing['waveform'].append((end_t - start_t) / num_tokens)

                out, timing_example = self.process_tokens(waveforms, target_sampling_rate, num_tokens)
                timing['spectrogram'].append(float(np.mean(timing_example)))
                timing['processing'].append(out['timing'])
    
                feats = out['feats'].cpu().numpy().astype(np.float32)
                embs = out['embs'].cpu().numpy().astype(np.float32)
                embs_pca = out['embs_pca'].cpu().numpy().astype(np.uint8)
                feats_spa_gpa = feats.mean(axis=(-1, -2))
        
                # Save datasets (uncomment if raw waveforms should be saved)
                # hf_waveforms.create_dataset(video_id, data=waveforms, dtype=np.float32, compression="gzip")

                hf_spectrograms.create_dataset(video_id, data=out['spectrograms'], dtype=np.float32, compression="gzip")
                hf_feats.create_dataset(video_id, data=feats, dtype=np.float32, compression="gzip")
                hf_feats_spa_gpa.create_dataset(video_id, data=feats_spa_gpa, dtype=np.float32, compression="gzip")
                hf_embs.create_dataset(video_id, data=embs, dtype=np.float32, compression="gzip")
                hf_embs_pca.create_dataset(video_id, data=embs_pca, dtype=np.uint8, compression="gzip")

        with open(output_timing_path, 'wb') as f:
            pickle.dump(timing, f)


def main():
    config = {
        "input_path": "data/AVE_trimmed",
        "output_waveforms_path": "data/raw/audio_waveforms.h5",
        "output_spectrograms_path": "data/raw/spectrograms/audio_cor96.h5",
        "output_features_path": "data/classification/features/audio.h5",
        "output_features_spa_gpa_path": "data/classification/features/audio_spa_gpa.h5",
        "output_embeddings_path": "data/classification/embeddings/audio.h5",
        "output_embeddings_pca_path": "data/classification/embeddings/audio_pca.h5",
        "output_timing_path": "data/classification/timing/audio_pipeline.pth",
        "target_sampling_rate": 16000,
        "num_tokens": 10,
        "token_duration": 1.0,
        "device": (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
            else torch.device('cpu')
        ),
        "seed": 42
    }

    set_seed(config['seed'])

    audio_wrapper = AudioPipelineWrapper(
        device=config['device'],
        pretrained=True,
        preprocess=False,
        postprocess=True
    )

    audio_wrapper.process_videos(
        input_path=config["input_path"],
        num_tokens=config["num_tokens"],
        token_duration=config["token_duration"],
        target_sampling_rate=config["target_sampling_rate"],
        output_waveforms_path=config["output_waveforms_path"],
        output_spectrograms_path=config["output_spectrograms_path"],
        output_features_path=config["output_features_path"],
        output_features_spa_gpa_path=config["output_features_spa_gpa_path"],
        output_embeddings_path=config["output_embeddings_path"],
        output_embeddings_pca_path=config["output_embeddings_pca_path"],
        output_timing_path=config["output_timing_path"],
    )


if __name__ == "__main__":
    main()
