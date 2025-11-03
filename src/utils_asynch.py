import torch
import numpy as np

from fractions import Fraction

from src.vggish_input import waveform_to_examples

# =================== Utility Functions ===================
def extract_streaming_audio_embs(audio_buffer, pipeline, sample_rate=16000, packet_duration=Fraction(1, 50), chunk_duration=1.0, device='cpu'):
    """
    
    """

    # Check if audio_buffer is empty
    if len(audio_buffer) == 0:
        #breakpoint()
        return {
            'num_chunks': 0,
            'num_complete_chunks': 0,
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

    # How many complete chunks we can have
    num_complete_chunks = int(waveform_duration / Fraction(chunk_duration))

    # How many chunks can we extract
    num_chunks = np.ceil(waveform_duration / Fraction(chunk_duration))

    # Compute number of excedding packets
    num_exceeding_pkts = 0
    if num_chunks > num_complete_chunks:
        exceeding_duration = (waveform_duration - num_complete_chunks * Fraction(chunk_duration))
        num_exceeding_pkts = int(exceeding_duration / packet_duration)

    # Compute number of non-exceeding packets
    num_non_exceeding_pkts = len(audio_buffer) - num_exceeding_pkts

    # Compute expected total length 
    expected_len = int(num_chunks * chunk_duration * sample_rate) 
    
    # Padding if necessary
    if current_len < expected_len:
        pad_width = expected_len - current_len
        waveform = np.pad(waveform, (0, pad_width), mode='constant', constant_values=0)
    elif current_len > expected_len:
        breakpoint() # TODO: handle this case

    # breakpoint()
    # # Prepare to figure out exceding packets if needed
    # exceeding_pkts = None

    # # More information than needed
    # if exceeding_len > 0:

    #     if mode == 'past':
    #         # Just truncatate waveform to expected length
    #         waveform = waveform[:expected_len]

    #     elif mode == 'no-discard':
    #         # Truncate waveform to expected length
    #         waveform = waveform[:expected_len]

    #         acc_len = 0
    #         count = 0

    #         # Iterate from the last packet backward, accumulate payload lengths
    #         for pkt in reversed(audio_buffer):
    #             acc_len += len(pkt.payload)
    #             count += 1
    #             if acc_len >= exceeding_len:
    #                 break
            
    #         # Select the last `count` packets as exceeding_pkts (not to discard)
    #         exceeding_pkts = list(audio_buffer)[-count:]  

    # Reshape waveform to (number of chunks, num_samples_per_chunk)
    waveform = waveform.reshape(num_chunks, -1)

    # Iterate over chunks and extract spectrograms
    spectrograms = []
    for cc in range(num_chunks):

        # Extract current chunk waveform
        waveform_chunk = waveform[cc]

        # Extract current spectrogram
        spectrogram = waveform_to_examples(data=waveform_chunk, sample_rate=sample_rate, return_tensor=False)

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
        'num_chunks': num_chunks,
        'num_complete_chunks': num_complete_chunks,
        'num_exceeding_pkts': num_exceeding_pkts,
        'num_non_exceeding_pkts': num_non_exceeding_pkts,
        'embs': embs
    }

    return out


def extract_streaming_video_feats(video_buffer, fallback_feature, sample_rate=16, packet_duration=Fraction(1, 16), chunk_duration=1.0, device='cpu'):
    """

    """

    # TODO: check if video_buffer is empty
    if len(video_buffer) == 0:
        breakpoint()
        return {
            'num_chunks': 0,
            'num_complete_chunks': 0,
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

    # How many complete chunks we can have
    num_complete_chunks = int(frame_duration / Fraction(chunk_duration))

    # How many chunks can we extract
    num_chunks = np.ceil(frame_duration / Fraction(chunk_duration))

    # Compute number of excedding packets
    num_exceeding_pkts = 0
    if num_chunks > num_complete_chunks:
        exceeding_duration = (frame_duration - num_complete_chunks * Fraction(chunk_duration))
        num_exceeding_pkts = int(exceeding_duration / packet_duration)
    
    # Compute number of non-exceeding packets
    num_non_exceeding_pkts = len(video_buffer) - num_exceeding_pkts

    # Compute expected total length 
    expected_len = int(num_chunks * chunk_duration * sample_rate) 

    # Padding if necessary
    if current_len < expected_len:
        pad_len = expected_len - len(feats)
        # Repeat fallback_feature pad_len times along the feature axis 0
        pad_feats = np.tile(fallback_feature, (pad_len, 1, 1, 1)) 
        # Concatenate existing feats with padding
        feats = np.concatenate((feats, pad_feats), axis=0)  
    elif current_len > expected_len:
        breakpoint() # TODO: handle this case

    
    # Take the temporal mean as a way of pooling over a chunk
    feats = feats.reshape(int(num_chunks), -1, feats.shape[1], feats.shape[2], feats.shape[3])
    feats = feats.mean(axis=1, keepdims=True)
    feats = feats.squeeze(1)  # (num_chunks, C, H, W)
    
    # Prepare output
    out = {
        'num_chunks': num_chunks,
        'num_complete_chunks': num_complete_chunks,
        'num_exceeding_pkts': num_exceeding_pkts,
        'num_non_exceeding_pkts': num_non_exceeding_pkts,
        'feats': feats
    }

    return out


def extract_fallback_audio_chunk_embs(pipeline, sample_rate=16000, chunk_duration=1.0, device='cpu'):
    """
    Extract a fallback audio chunk embedding from silent audio.

    """
    
    # Calculate expected length
    expected_len = int(chunk_duration * sample_rate)

    # Generate a silent waveform
    waveform = np.zeros(expected_len)

    # Extract mel-spectrogram example from this silent waveform
    spectrogram = waveform_to_examples(data=waveform, sample_rate=sample_rate, return_tensor=False)

    # Convert to tensor and send to device
    spectrogram = torch.tensor(spectrogram, device=device, dtype=torch.float32)

    # Generate fallback feature from silent audio over a chunk
    with torch.no_grad():
        embs = pipeline(spectrogram.unsqueeze(0), return_embs=True)['embs']

    # Convert to numpy
    embs = embs.cpu().numpy().astype(np.float32)

    return embs


def has_order_violation(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return True  # Order violation found
    return False  # No violations, array is sorted