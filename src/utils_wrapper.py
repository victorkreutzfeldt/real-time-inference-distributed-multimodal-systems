# src/utils_wrapper.py

import torch
import numpy as np

from fractions import Fraction

from src.vggish_input import waveform_to_examples


def extract_streaming_audio_embs(audio_buffer, pipeline, sample_rate=16000, packet_duration=Fraction(1, 50), token_duration=1.0, device='cpu'):
    """
    
    """

    # Check if audio_buffer is empty
    if len(audio_buffer) == 0:
        return {
            'num_tokens': 0,
            'num_complete_tokens': 0,
            'num_exceeding_pkts': 0,
            'num_non_exceeding_pkts': 0,
            'embs': np.zeros((0, 128), dtype=np.float32)
        }

    # Retrieve waveform from packets' payloads
    waveform = [pkt.payload for pkt in audio_buffer]
    waveform = np.concatenate(waveform, axis=0)
    
    # Squeeze waveform
    waveform = np.squeeze(waveform)

    # Waveform current length
    current_len = len(waveform)

    # Waveform duration
    waveform_duration = Fraction(current_len, sample_rate)

    # How many complete tokens we can have
    num_complete_tokens = int(waveform_duration / Fraction(token_duration))

    # How many tokens can we extract
    num_tokens = np.ceil(waveform_duration / Fraction(token_duration))

    # Compute number of excedding packets
    num_exceeding_pkts = 0
    if num_tokens > num_complete_tokens:
        exceeding_duration = (waveform_duration - num_complete_tokens * Fraction(token_duration))
        num_exceeding_pkts = int(exceeding_duration / packet_duration)

    # Compute number of non-exceeding packets
    num_non_exceeding_pkts = len(audio_buffer) - num_exceeding_pkts

    # Compute expected total length 
    expected_len = int(num_tokens * token_duration * sample_rate) 
    
    # Padding if necessary
    if current_len < expected_len:
        pad_width = expected_len - current_len
        waveform = np.pad(waveform, (0, pad_width), mode='constant', constant_values=0)
    elif current_len > expected_len:
        breakpoint() # TODO: handle this case

    # Reshape waveform to (number of tokens, num_samples_per_token)
    waveform = waveform.reshape(num_tokens, -1)

    # Iterate over tokens and extract spectrograms
    spectrograms = []
    for cc in range(num_tokens):

        # Extract current token waveform
        waveform_token = waveform[cc]

        # Extract current spectrogram
        spectrogram = waveform_to_examples(data=waveform_token, sample_rate=sample_rate, return_tensor=False)

        # Store current spectrogram
        spectrograms.append(spectrogram)

    # Convert to tensor and send to device
    spectrograms = np.stack(spectrograms, axis=0)
    spectrograms = torch.tensor(spectrograms, device=device, dtype=torch.float32)

    # Extract embeddings
    with torch.no_grad():
        embs = pipeline(spectrograms, return_embs=True)['embs'] 

    # Convert to numpy
    embs = embs.cpu().numpy().astype(np.float32)

    # Prepare output
    out = {
        'num_tokens': num_tokens,
        'num_complete_tokens': num_complete_tokens,
        'num_exceeding_pkts': num_exceeding_pkts,
        'num_non_exceeding_pkts': num_non_exceeding_pkts,
        'embs': embs
    }

    return out


def extract_streaming_video_feats(video_buffer, fallback_feature, sample_rate=16, packet_duration=Fraction(1, 16), token_duration=1.0, device='cpu'):
    """

    """

    # Check if video_buffer is empty
    if len(video_buffer) == 0:
        breakpoint()
        return {
            'num_tokens': 0,
            'num_complete_tokens': 0,
            'num_exceeding_pkts': 0,
            'num_non_exceeding_pkts': 0,
            'feats': np.zeros((0, fallback_feature.shape[1], fallback_feature.shape[2], fallback_feature.shape[3]), dtype=np.float32)
        }

    # Retrieve feats from packets' payloads
    feats = [pkt.payload for pkt in video_buffer]
    feats = np.stack(feats, axis=0)  # (N, C, H, W)

    # Extract current number of frames
    current_len = len(video_buffer)

    # Frame duration
    frame_duration = Fraction(current_len, sample_rate)

    # How many complete tokens we can have
    num_complete_tokens = int(frame_duration / Fraction(token_duration))

    # How many tokens can we extract
    num_tokens = np.ceil(frame_duration / Fraction(token_duration))

    # Compute number of excedding packets
    num_exceeding_pkts = 0
    if num_tokens > num_complete_tokens:
        exceeding_duration = (frame_duration - num_complete_tokens * Fraction(token_duration))
        num_exceeding_pkts = int(exceeding_duration / packet_duration)
    
    # Compute number of non-exceeding packets
    num_non_exceeding_pkts = len(video_buffer) - num_exceeding_pkts

    # Compute expected total length 
    expected_len = int(num_tokens * token_duration * sample_rate) 

    # Padding if necessary
    if current_len < expected_len:
        pad_len = expected_len - len(feats)
        # Repeat fallback_feature pad_len times along the feature axis 0
        pad_feats = np.tile(fallback_feature, (pad_len, 1, 1, 1)) 
        # Concatenate existing feats with padding
        feats = np.concatenate((feats, pad_feats), axis=0)  
    elif current_len > expected_len:
        breakpoint() # TODO: handle this case

    
    # Take the temporal mean as a way of pooling over a token
    feats = feats.reshape(int(num_tokens), -1, feats.shape[1], feats.shape[2], feats.shape[3])
    feats = feats.mean(axis=1, keepdims=True)
    feats = feats.squeeze(1)  # (num_tokens, C, H, W)
    
    # Prepare output
    out = {
        'num_tokens': num_tokens,
        'num_complete_tokens': num_complete_tokens,
        'num_exceeding_pkts': num_exceeding_pkts,
        'num_non_exceeding_pkts': num_non_exceeding_pkts,
        'feats': feats
    }

    return out


def extract_fallback_audio_token_embs(pipeline, sample_rate=16000, token_duration=1.0, device='cpu'):
    """
    Extract a fallback audio token embedding from silent audio.

    """
    
    # Calculate expected length
    expected_len = int(token_duration * sample_rate)

    # Generate a silent waveform
    waveform = np.zeros(expected_len)

    # Extract mel-spectrogram example from this silent waveform
    spectrogram = waveform_to_examples(data=waveform, sample_rate=sample_rate, return_tensor=False)

    # Convert to tensor and send to device
    spectrogram = torch.tensor(spectrogram, device=device, dtype=torch.float32)

    # Generate fallback feature from silent audio over a token
    with torch.no_grad():
        embs = pipeline(spectrogram.unsqueeze(0), return_embs=True)['embs']

    # Convert to numpy
    embs = embs.cpu().numpy().astype(np.float32)

    return embs