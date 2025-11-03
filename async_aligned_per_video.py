import gzip
import os
import pickle
import random
import time
from typing import List
from collections import deque

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fractions import Fraction

from src.communication import simulate_transmission
from src.datasets import PerVideoMultimodalDatasetLabels

from src._class_pipeline_audio import AudioPipeline
from src._class_pipeline_video import VideoPipeline

from src.packets import load_transmitted_packets_from_saved_features
from src.per_video_models import PerVideoBiLSTMMultimodalClassifier

from src.utils import hamming_accuracy_from_label_lists, subset_accuracy_from_label_lists
from src.utils_asynch import extract_fallback_audio_chunk_embs, extract_streaming_audio_embs, extract_streaming_video_feats

# ================ Config =====================
ANNOTATIONS_CSV = 'data/annotations.csv'

# Data parameters
VIDEO_DURATION = 10.0  # in seconds
NUM_CHUNKS = 10
CHUNK_DURATION = 1.0
CHUNKS_PER_VIDEO = 10
NUM_CLASSES = 29
AUDIO_SAMPLING_RATE = 16_000
VIDEO_FPS = 16

# # Network parameters

def rate(epsilon, bandwidth, snr_db):
    snr_linear = 10**(snr_db/10)
    val = 1 - snr_linear * np.log(1 - epsilon)
    val = np.maximum(val, 1e-12)  # prevent log domain error
    return bandwidth * np.log2(val)

# Bandwidth in Hz
AUDIO_BANDWIDTH = 1.08e6
VIDEO_BANDWIDTH = 100e6

# Outage probability
outage_proba = 0.5

# SNR in dB
AUDIO_SNR_DB = 2
VIDEO_SNR_DB = 0 

BANDWIDTH_BPS_AUDIO = rate(outage_proba, AUDIO_BANDWIDTH, AUDIO_SNR_DB)
BANDWIDTH_BPS_VIDEO = rate(outage_proba, VIDEO_BANDWIDTH, VIDEO_SNR_DB)

# LOSS_PROB_AUDIO = 0.0
# LOSS_PROB_VIDEO = 0.0

# LATENCY = 0.05
# JITTER = 0.02

# Transmission parameters
AUDIO_PACKET_SIZE = 5_120 # bits
VIDEO_PACKET_SIZE = 1_204_224 # bits

AUDIO_PACKET_DURATION = Fraction(1, 50)
VIDEO_PACKET_DURATION = Fraction(1, 16)

#AUDIO_TX_DELAY = #Fraction(AUDIO_PACKET_SIZE, BANDWIDTH_BPS_AUDIO)
#VIDEO_TX_DELAY = #Fraction(VIDEO_PACKET_SIZE, BANDWIDTH_BPS_VIDEO)

# Asynch parameters
WINDOW_DURATION = float(np.max((50 * (4.4331/ 500), 16 * (5.0724 / 160))))
WINDOW_DURATION = np.ceil(WINDOW_DURATION * 10000) / 10000

#max(AUDIO_TX_DELAY, VIDEO_TX_DELAY)  # step: 100ms

STOP_TIME = 4.4331

#WINDOWS_PER_CHUNK = int(CHUNK_DURATION / WINDOW_DURATION)  # 10 windows = 1 second
#MODE = 'no-discard'

# Inference parameters
MODEL_CHECKPOINT = 'models/classification/per_video/shallow_classifier_multimodal_features_base.pth'

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    else torch.device('cpu')
)

# Output
SAVE_DIR = "data/communication/async"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f'per_video_aligned_deterministic_SNR_{AUDIO_SNR_DB}.gz')

# ================ Logging =====================
#print(f"[INFO] Mode: {MODE}")
print(f"[INFO] Window duration: {float(WINDOW_DURATION):.3f}s")
#print(f"[INFO] Loss probability: {LOSS_PROB:.2f}")

# ================ Main =====================
if __name__ == "__main__":
    
    # Load test dataset
    test_ds = PerVideoMultimodalDatasetLabels(ANNOTATIONS_CSV, 'test', NUM_CHUNKS, NUM_CLASSES)
    test_videos = test_ds.video_ids

    # Get number of test videos
    num_test_videos = len(test_videos)

    # Get audio and video feature extractors
    pipeline_audio = AudioPipeline(device=DEVICE)
    pipeline_audio.eval()

    pipeline_video = VideoPipeline(device='cpu', preprocess=True)
    pipeline_video.eval()

    # Fallback feature for missing audio chunk
    fallback_audio_chunk_embs = extract_fallback_audio_chunk_embs(
        pipeline=pipeline_audio, sample_rate=AUDIO_SAMPLING_RATE, chunk_duration=CHUNK_DURATION, device=DEVICE)
    
    # Get feature for a dark image
    dark_image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    with torch.no_grad():
        out = pipeline_video(dark_image, return_feats=True) 
    del pipeline_video
    fallback_video_feat = out['feats'].squeeze(0).numpy().astype(np.float32)  

    # Get inference model
    model = PerVideoBiLSTMMultimodalClassifier().to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))

    # Prepare to save results    
    results = {
        'hamming_acc': [[] for _ in range(num_test_videos)],
        'subset_acc': [[] for _ in range(num_test_videos)],
        #'causal_hamming_acc': [[] for _ in range(num_test_videos)],
        #'causal_subset_acc': [[] for _ in range(num_test_videos)],
        'num_rx_pkts_audio': [[] for _ in range(num_test_videos)],
        'num_rx_pkts_video': [[] for _ in range(num_test_videos)],
        'curr_num_completed_chunks_audio': [[] for _ in range(num_test_videos)],
        'curr_num_completed_chunks_video': [[] for _ in range(num_test_videos)],
        #'curr_samples_audio': [[] for _ in range(num_test_videos)],
        #'curr_samples_video': [[] for _ in range(num_test_videos)],
        'num_missed_pkts_audio': [None for _ in range(num_test_videos)],
        'num_missed_pkts_video': [None for _ in range(num_test_videos)]
    }

    # Save metadata
    results['metadata'] = {
        #'ANNOTATIONS_CSV': ANNOTATIONS_CSV,
        #'VIDEO_DURATION': VIDEO_DURATION,
        #'NUM_CHUNKS': NUM_CHUNKS,
        #'CHUNK_DURATION': CHUNK_DURATION,
        #'AUDIO_SAMPLING_RATE': AUDIO_SAMPLING_RATE,
        #'VIDEO_FPS': VIDEO_FPS,
        'WINDOW_DURATION': WINDOW_DURATION,
        'BANDWIDTH_BPS_AUDIO': BANDWIDTH_BPS_AUDIO,
        'BANDWIDTH_BPS_VIDEO': BANDWIDTH_BPS_VIDEO,
        #'LATENCY': LATENCY,
        #'JITTER': JITTER,
        # 'LOSS_PROB_AUDIO': LOSS_PROB_AUDIO,
        # 'LOSS_PROB_VIDEO': LOSS_PROB_VIDEO,
        'AUDIO_PACKET_SIZE': AUDIO_PACKET_SIZE,
        'VIDEO_PACKET_SIZE': VIDEO_PACKET_SIZE,
        'NUM_WINDOWS': [None for _ in range(num_test_videos)],
        #'TIME': time.strftime("%Y-%m-%d %H:%M:%S"), # timestamp for reproducibility
    }

    # Go through all test videos
    for idx, video_id in tqdm(enumerate(test_videos), total=len(test_videos), desc="Aligned Async Inference", ascii=True):

        # Extract labels from the dataset
        data = test_ds[idx]
        labels = data['labels']

        # Get audio and video base names
        audio_base = f"data/packets/audio/{video_id}"
        video_base = f"data/packets/video/{video_id}"

        # Load TO-BE transmitted packets
        transmitted_audio = load_transmitted_packets_from_saved_features(
            packets_path=f"{audio_base}_packets.pkl.gz",
            payload_path=f"{audio_base}_payloads.h5",
            pts2idx_path=f"{audio_base}_pts2idx.pkl.gz",
            stream_type='audio'
        )

        transmitted_video = load_transmitted_packets_from_saved_features(
            packets_path=f"{video_base}_packets.pkl.gz",
            payload_path=f"{video_base}_payloads.h5",
            pts2idx_path=f"{video_base}_pts2idx.pkl.gz",
            stream_type='video'
        )

        # Simulate transmission
        received_audio = simulate_transmission(transmitted_audio, BANDWIDTH_BPS_AUDIO, outage_proba)
        received_video = simulate_transmission(transmitted_video, BANDWIDTH_BPS_VIDEO, outage_proba, fallback_video_feature=fallback_video_feat)
        
        # Get all PTS times
        pts_times_audio = [float(pkt.pts_time) for pkt in received_audio]
        pts_times_video = [float(pkt.pts_time) for pkt in received_video]

        # Get all arrival times
        arrival_times_audio = [float(pkt.arrival_time) for pkt in received_audio if pkt.arrival_time is not None]
        arrival_times_video = [float(pkt.arrival_time) for pkt in received_video if pkt.arrival_time is not None]

        # Send received audio and video to buffer
        received_audio_buffer = deque(received_audio)
        received_video_buffer = deque(received_video)

        # Create audio and video buffers
        audio_buffer = deque()
        video_buffer = deque()
        
        # Create audio and video buffers per chunk
        audio_chunks_buffer = deque()
        video_chunks_buffer = deque()

        # Initalize number of complete and incomplete chunks
        num_complete_chunks_audio = 0
        num_incomplete_chunks_audio = CHUNKS_PER_VIDEO

        num_complete_chunks_video = 0
        num_incomplete_chunks_video = CHUNKS_PER_VIDEO

        # Window control
        ww = 0  # window counter
        PROCESS_AUDIO = True
        PROCESS_VIDEO = True
        last_round = None

        # Go over windows
        while True:


            ########## Window Buffers ##########  


            # Get start and end times of this window
            t_start = ww * WINDOW_DURATION
            t_end = t_start + WINDOW_DURATION 
            
            # Store current number of received packets 
            window_audio = []
            while PROCESS_AUDIO and received_audio_buffer and t_start < received_audio_buffer[0].arrival_time <= t_end:
                pkt = received_audio_buffer.popleft()
                window_audio.append(pkt)

            window_video = []
            while PROCESS_VIDEO and received_video_buffer and t_start < received_video_buffer[0].arrival_time <= t_end:
                pkt = received_video_buffer.popleft()
                window_video.append(pkt)
           
            # Get the number of received packets in this window and fill buffers
            if PROCESS_AUDIO:
                num_rx_pkts_audio = len(window_audio)
                results['num_rx_pkts_audio'][idx].append(num_rx_pkts_audio)
                audio_buffer.extend(window_audio)
            else:
                num_rx_pkts_audio = 0
                results['num_rx_pkts_audio'][idx].append(num_rx_pkts_audio) 

            if PROCESS_VIDEO:
                num_rx_pkts_video = len(window_video)
                results['num_rx_pkts_video'][idx].append(num_rx_pkts_video)
                video_buffer.extend(window_video)
            else:
                num_rx_pkts_video = 0
                results['num_rx_pkts_video'][idx].append(num_rx_pkts_video) 
            #breakpoint()

            ########## Alignment ##########

            #breakpoint()
            # Perfect alignment based on PTS (reference time of both streams are synched)
            if PROCESS_AUDIO:
                audio_buffer = deque(sorted(audio_buffer, key=lambda p: p.pts_time))

            if PROCESS_VIDEO:
                video_buffer = deque(sorted(video_buffer, key=lambda p: p.pts_time))


            ######### Feature Extraction ##########


            # Apply preprocessing pipeline considering buffered data
            if PROCESS_AUDIO:
                out_audio = extract_streaming_audio_embs(audio_buffer, pipeline_audio)
            
            if PROCESS_VIDEO:
                out_video = extract_streaming_video_feats(video_buffer, fallback_video_feat)

            # # Store current number of samples
            # results['curr_samples_audio'][idx, ww] = curr_num_audio_samples
            # results['curr_samples_video'][idx, ww] = curr_num_video_samples

            # # Duration covered by audio and video
            # results['curr_duration_audio'][idx, ww] = sum([pkt.duration for pkt in audio_windows_buffer])
            # results['curr_duration_video'][idx, ww] = sum([pkt.duration for pkt in video_windows_buffer])


            # ########## Handle Current Data ########## 


            # Check if we have completed chunk information
            if PROCESS_AUDIO and out_audio['num_complete_chunks'] > 0: 
                audio_chunks_buffer.extend(out_audio['embs'][:out_audio['num_complete_chunks']])
                num_complete_chunks_audio += out_audio['num_complete_chunks']
                num_incomplete_chunks_audio = CHUNKS_PER_VIDEO - num_complete_chunks_audio

            if PROCESS_VIDEO and out_video['num_complete_chunks'] > 0: 
                video_chunks_buffer.extend(out_video['feats'][:out_video['num_complete_chunks']])
                num_complete_chunks_video += out_video['num_complete_chunks']
                num_incomplete_chunks_video = CHUNKS_PER_VIDEO - num_complete_chunks_video

            # Store current number of completed chunks
            results['curr_num_completed_chunks_audio'][idx].append(num_complete_chunks_audio)
            results['curr_num_completed_chunks_video'][idx].append(num_complete_chunks_video)

            # Get current partial complete chunk (if any)
            curr_partially_complete_audio = None
            if PROCESS_AUDIO and out_audio['num_exceeding_pkts'] > 0:
                curr_partially_complete_audio = out_audio['embs'][out_audio['num_complete_chunks']:out_audio['num_complete_chunks'] + 1]

            curr_partially_complete_video = None
            if PROCESS_VIDEO and out_video['num_exceeding_pkts'] > 0:
                curr_partially_complete_video = out_video['feats'][out_video['num_complete_chunks']:out_video['num_complete_chunks'] + 1]


            # Get current data - audio
            if PROCESS_AUDIO:
                if num_complete_chunks_audio > 0:
                    if curr_partially_complete_audio is not None:
                        curr_audio = np.concatenate((list(audio_chunks_buffer), curr_partially_complete_audio), axis=0)
                    else:
                        curr_audio = np.stack(audio_chunks_buffer, axis=0)
                else:
                    # no complete chunks yet
                    if curr_partially_complete_audio is not None:
                        curr_audio = curr_partially_complete_audio
                    else:
                        # both empty, assign empty array explicitly if needed
                        curr_audio = np.empty((0, *audio_chunks_buffer[0].shape[1:]))  # TODO: adjust shape accordingly

            # Get current data - video
            if PROCESS_VIDEO:
                if num_complete_chunks_video > 0:
                    if curr_partially_complete_video is not None:
                        curr_video = np.concatenate((list(video_chunks_buffer), curr_partially_complete_video), axis=0)
                    else:
                        curr_video = np.stack(video_chunks_buffer, axis=0)
                else:
                    if curr_partially_complete_video is not None:
                        curr_video = curr_partially_complete_video
                    else:
                        curr_video = np.empty((0, *video_chunks_buffer[0].shape[1:]))  # TODO: adjust shape accordingly
           

            # ########## Handle Future Data ########## 


            # Get number of partially complete and complete chunks and complete the rest with fallback features
            if PROCESS_AUDIO:
                num_partially_complete_chunks_audio = (1 if out_audio['num_exceeding_pkts'] > 0 else 0)
            
                # Concatenate incomplete chunks
                if num_incomplete_chunks_audio > 0:
                    incomplete_chunks_audio_embs = np.tile(fallback_audio_chunk_embs, (num_incomplete_chunks_audio-num_partially_complete_chunks_audio, 1))
                    curr_audio = np.concatenate((curr_audio, incomplete_chunks_audio_embs), axis=0)
                else:
                    curr_audio = curr_audio

            if PROCESS_VIDEO:
                num_partially_complete_chunks_video = (1 if out_video['num_exceeding_pkts'] > 0 else 0)
            
                # Concatenate incomplete chunks
                if num_incomplete_chunks_video > 0:
                    incomplete_chunks_video_feat = np.tile(fallback_video_feat, (num_incomplete_chunks_video-num_partially_complete_chunks_video, 1, 1, 1))
                    curr_video = np.concatenate((curr_video, incomplete_chunks_video_feat), axis=0)
                else:
                    curr_video = curr_video

            
            # ########## Inference ##########
            

            # Convert to tensor, send to device, and unsequeeze to enable inference
            audio_tensor = torch.tensor(curr_audio, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            video_tensor = torch.tensor(curr_video, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            # TODO: check labels 
            
            # Perform inference
            with torch.no_grad():
                try:
                    logits = model(audio_tensor, video_tensor)
                except:
                    breakpoint()
            
            # Compute probabilities
            probas = torch.sigmoid(logits).squeeze(0)

            # Compute predictions
            preds = (probas > 0.5)

            # Transform into lists
            predicted_labels = [torch.nonzero(row).squeeze(1).tolist() for row in preds]
            true_labels = [torch.nonzero(row).squeeze(1).tolist() for row in labels]
        
            # Compute hamming and subset accuracies
            hamming_acc = hamming_accuracy_from_label_lists(true_labels, predicted_labels)
            subset_acc = subset_accuracy_from_label_lists(true_labels, predicted_labels)

            # Compute causal hamming and subset accuracies
            #causal_hamming_acc = hamming_accuracy_from_label_lists(true_labels[:num_complete_chunks + 1], predicted_labels[:num_complete_chunks + 1])
            #causal_subset_acc = subset_accuracy_from_label_lists(true_labels[:num_complete_chunks + 1], predicted_labels[:num_complete_chunks + 1]) 

            # Store accuracies
            results['hamming_acc'][idx].append(float(hamming_acc))
            results['subset_acc'][idx].append(float(subset_acc))
            #breakpoint()
            # results['causal_hamming_acc'][idx, ww] = float(causal_hamming_acc)
            # results['causal_subset_acc'][idx, ww] = float(causal_subset_acc)


            ########## Next Window ########## 

            if t_end >= STOP_TIME:
                break 

            # Keep exceeding packets in the buffer
            if PROCESS_AUDIO and out_audio['num_exceeding_pkts'] > 0:
                audio_buffer = deque(list(audio_buffer)[-out_audio['num_exceeding_pkts']:])
            else:
                audio_buffer.clear()

            if PROCESS_VIDEO and out_video['num_exceeding_pkts'] > 0:
                video_buffer = deque(list(video_buffer)[-out_video['num_exceeding_pkts']:])
            else:
                video_buffer.clear()

            # Should we stop processing any modality?
            if num_complete_chunks_audio == CHUNKS_PER_VIDEO:
                PROCESS_AUDIO = False

            if num_complete_chunks_video == CHUNKS_PER_VIDEO:
                PROCESS_VIDEO = False

            # If both modalities are done, we can stop after this round
            if not PROCESS_AUDIO and not PROCESS_VIDEO:
                last_round = ww

            # Increment window counter
            ww += 1

            # Stopping condition
            if last_round is not None and ww > last_round:
                break

        # Store number of missed packets
        results['num_missed_pkts_audio'] = len(received_audio_buffer)
        results['num_missed_pkts_video'] = len(received_video_buffer)
    
        # Store number of windows processed
        results['metadata']['NUM_WINDOWS'][idx] = ww

    # Save results
    with gzip.open(SAVE_PATH, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)