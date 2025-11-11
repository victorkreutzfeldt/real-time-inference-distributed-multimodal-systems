# src/communication.py

"""
Module for communication channel modeling and packet transmission simulation.

Provides functions such as `rate` to compute achievable bit rates given channel conditions, and 
`simulate_transmission` to emulate sequential packet transmission over lossy channels by updating 
packet delays and arrival times.

Author: Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
Date: 2025-11-11
"""

import numpy as np

from typing import List

from src.packets import TransmittedPacket


def rate(snr_dB: float, bandwidth: float, outage_proba: float) -> float:
    """
    Compute the achievable rate using the erasure channel model specified in the paper.

    Args:
        snr_dB (float): Signal-to-noise ratio in dB.
        bandwidth (float): Channel bandwidth in Hz.
        outage_proba (float): Outage probability (between 0 and 1).

    Returns:
        float: Achievable rate in bits per second.
    """
    snr_linear = 10 ** (snr_dB / 10)
    val = 1 - snr_linear * np.log(1 - outage_proba)
    val = np.maximum(val, 1e-12)  # prevent log domain error
    rate = float((bandwidth * np.log2(val)).item())

    return rate


def simulate_transmission(stream: List[TransmittedPacket], config: dict, modality: str) -> List[TransmittedPacket]:
    """
    Simulate the transmission of packets over a channel with given bandwidth and outage probability. 
    Packets are transmitted sequentially, and their arrival times are computed based on their sizes and the channel conditions.

    Args:
        stream (List[TransmittedPacket]): List of packets to be transmitted.
        config (dict): Configuration dictionary containing channel parameters.
        modality (str): 'audio' or 'video' to select the appropriate channel parameters.
  
    Returns:
        List[TransmittedPacket]: List of packets with updated transmission delays and arrival times.
    """

    # List to store received packets
    received = []

    # Initialize last arrival time
    last_arrival_time = 0.0

    # Extract parameters from config
    snr_dB = config['modalities'][modality]['snr_dB']
    bandwidth = config['modalities'][modality]['bandwidth']
    outage_proba = config['modalities'][modality]['outage_proba']

    # Compute achievable rate
    bandwidth_bps = rate(snr_dB, bandwidth, outage_proba)

    # Iterate over packets and simulate transmission
    for pkt in stream:
       
        # Compute and store transmission delay based on average statistics
        tx_delay = (pkt.size_bits / float(bandwidth_bps)) / (1 - outage_proba)
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