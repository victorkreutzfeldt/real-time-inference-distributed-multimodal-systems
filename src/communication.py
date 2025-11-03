import math
import torch
import numpy as np

from typing import List

from src.packets import TransmittedPacket
from fractions import Fraction

def apply_clock_drift(packets, initial_drift=0.0, drift_per_sec=0.01, fallback_feature=None):
    """
    Applies linear clock drift to a stream of packets and fills the
    initial drift gap with fallback packets.

    Args:
        packets: list of TransmittedPacket (sorted by pts_time)
        initial_drift: starting drift offset in seconds
        drift_per_sec: drift to add per second of media time (s/s)
        fallback_feature: payload to use for synthetic fallback packets
        Other keyword args: used to create fallback packets
    Returns:
        List of TransmittedPacket with drift applied and initial drift packets prepended as needed.
    """
    # Determine per-packet duration
    #if not packets:
    #    return []
    #if duration is None:
    #    duration = packets[0].duration  # seconds per packet

    # Apply drift to each packet and mark drifted pts
    for pkt in packets:
        drift = initial_drift + pkt.pts_time * drift_per_sec
        pkt.pts_time_drifted = pkt.pts_time + drift

        # # Optionally apply drift to arrival_pts_time as well
        # if pkt.arrival_pts_time is not None:
        #     pkt.arrival_pts_time += drift

    # If there's initial drift, fill with fallback packets so that the drifted stream covers the full timeline
    fallback_packets = []
    if initial_drift > 0.0:

        # How many initial packets are needed to fill the drift gap?       
        n_fallback = int(math.ceil(initial_drift / packets[0].duration))
        
        for i in range(n_fallback):
            fallback_pts_time = packets[0].pts_time - (n_fallback - i) * packets[0].duration
            fallback_pkt = TransmittedPacket(
                stream_type=packets[0].stream_type,
                pts=None,
                pts_time=fallback_pts_time,
                duration=packets[0].duration,
                size_bits=packets[0].size_bits,
                sampling_rate=packets[0].sampling_rate,
                fps=packets[0].fps,
                nb_channels=packets[0].nb_channels,
                resolution=packets[0].resolution,
                payload=fallback_feature,
                tx_delay=0.0,
                arrival_pts_time=None,
                is_lost=False
            )

            # Mark drifted pts_time for the fallback
            fallback_pkt.pts_time_drifted = fallback_pkt.pts_time + initial_drift  # Only initial drift
            fallback_packets.append(fallback_pkt)

    # Get number of fallback packets
    num_fallback_packets = len(fallback_packets)

    # Concatenate fallback packets and the drifted original packets
    all_packets = fallback_packets + packets
    
    return all_packets, num_fallback_packets



#def simulate_transmission(packets: List[TransmittedPacket], bandwidth_bps, latency, jitter, loss_prob, fallback_video_feature: np.ndarray = None):
def simulate_transmission(packets: List[TransmittedPacket], bandwidth_bps, outage_proba, fallback_video_feature: np.ndarray = None):
    """
    
    """
    received = []
    last_arrival_time = 0.0

    # Iterate over packets and simulate transmission
    for pkt in packets:

        # # Simulate loss
        # if np.random.rand() < loss_prob:
        #     pkt.is_lost = True

            # if pkt.stream_type == 'audio':
            #     pkt.payload = np.zeros_like(pkt.payload)
                
            # elif pkt.stream_type == 'video':
            #     pkt.payload = fallback_video_feature.copy()
       
        # Compute and store transmission delay
        tx_delay = (pkt.size_bits / float(bandwidth_bps)) / (1 - outage_proba) #+ latency #+ np.random.uniform(0, jitter)
        pkt.tx_delay = tx_delay

        # Compute and store arrival time (cumulative)
        pkt.arrival_time = last_arrival_time + tx_delay
    
        # Update last arrival time
        last_arrival_time = pkt.arrival_time

        # Append to received 
        received.append(pkt)

    # Sort by arrival time (None at the end)
    received = sorted(received, key=lambda p: (p.arrival_time is None, p.arrival_time))

    return received 


# def reconstruct_audio_video_from_received_packets(
#     received_packets: List[TransmittedPacket],
#     expected_duration: float,
#     pad_missing: bool = True,
#     
# ):
#     """
#     Reconstructs the audio waveform and video frames from received packets.

#     Args:
#         received_packets (List[TransmittedPacket]): List of received packets (audio and video).
#         expected_duration (float): Expected media duration in seconds.
#         pad_missing (bool): Whether to pad missing segments (silence or black frames).

#     Returns:
#         Tuple[np.ndarray, Tuple[int, int], List[np.ndarray]]:
#             - audio waveform as (N,)
#             - list of video frames as np.ndarray
#     """
#     # Separate streams
#     packets_audio = [p for p in received_packets if p.stream_type == 'audio']
#     packets_video = [p for p in received_packets if p.stream_type == 'video']

#     # Sort by pts
#     packets_audio.sort(key=lambda x: x.pts)
#     packets_video.sort(key=lambda x: x.pts)

#     # TODO: extract sample rate and resolution
#     sampling_rate = packets_audio[0].sampling_rate
#     resolution = packets_video[0].resolution

#     # Audio reconstruction
#     stream_audio = []
#     cur_time = 0.0
#     for pkt in packets_audio:
#         if pad_missing and pkt.pts_time - cur_time > pkt.duration:
#             gap_duration = pkt.pts_time - cur_time
#             num_missing = int(gap_duration / pkt.duration)

#             breakpoint()

#             for _ in range(num_missing): 
#                 stream_audio.append(np.zeros((pad_len, pkt.channels), dtype=np.float32))

#         stream_audio.append(pkt.payload)
#         cur_time = max(cur_time, pkt.pts_time + pkt.duration)

#     if pad_missing and cur_time < expected_duration:
#         pad_len = int((expected_duration - cur_time) * sampling_rate)
#         stream_audio.append(np.zeros((pad_len, pkt.channels), dtype=np.float32))
#     breakpoint()
#     stream_audio = np.concatenate(stream_audio, axis=0)
     
#     # Video reconstruction
#     stream_video = []
#     cur_time = 0.0
#     for pkt in packets_video:
#         if pad_missing and pkt.pts_time - cur_time > pkt.duration:
#             gap_duration = pkt.pts_time - cur_time
#             num_missing = int(gap_duration / pkt.duration)
#             for _ in range(num_missing):
#                 stream_video.append(fallback_video_feature.copy())
#         stream_video.append(pkt.payload)
#         cur_time = pkt.pts_time + pkt.duration
#     if pad_missing and cur_time < expected_duration:
#         last_duration = pkt.duration if len(packets_video) > 0 else 1 / 16
#         num_missing = int((expected_duration - cur_time) / last_duration)
#         for _ in range(num_missing):
#             stream_video.append(fallback_video_feature.copy())
#     stream_video = np.stack(stream_video)
#     breakpoint()
#     return stream_audio, stream_video


# def reconstruct_stream(packets, duration=10.0, bin_size=1.0, stream_type='audio'):
#     num_bins = int(duration // bin_size)
#     bins = [None] * num_bins

#     for pkt in packets:
#         if pkt.stream_type != stream_type or pkt.pts_time is None:
#             continue
#         idx = int(pkt.pts_time // bin_size)
#         if 0 <= idx < num_bins:
#             bins[idx] = pkt.payload

#     # Fill missing bins with silence or black
#     for i in range(num_bins):
#         if bins[i] is None:
#             if stream_type == 'audio':
#                 shape = packets[0].payload.shape
#                 bins[i] = np.zeros_like(packets[0].payload)
#             elif stream_type == 'video':
#                 shape = packets[0].payload.shape
#                 bins[i] = np.zeros_like(packets[0].payload)
#     return bins
