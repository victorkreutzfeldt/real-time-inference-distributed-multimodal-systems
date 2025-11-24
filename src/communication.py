# src/communication.py

"""
Module for communication channel modeling and packet transmission simulation.

Provides functions such as `rate` to compute achievable bit rates given channel conditions, and 
`simulate_transmission` to emulate sequential packet transmission over lossy channels by updating 
packet delays and arrival times.

@author Victor Kreützfeldt (@victorkreutzfeldt)
Date: 2025-11-11
"""

from typing import List
from fractions import Fraction

import numpy as np

from src.packetization import Packet


def rate(snr_dB: float, bandwidth: float, outage_proba: float) -> float:
    """
    Compute the achievable rate using an erasure channel model, as specified in the paper.

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


def simulate_transmission(stream: List[Packet], config: dict, modality: str) -> List[Packet]:
    """
    Simulate the transmission of packets over a channel with given bandwidth and outage probability. 
    Packets are transmitted sequentially, and their arrival times are computed based on their sizes and the channel conditions.

    Args:
        stream (List[Packet]): List of packets to be transmitted.
        config (dict): Configuration dictionary containing channel parameters per modality.
        modality (str): 'audio' or 'video' to select the appropriate channel parameters.
  
    Returns:
        List[Packet]: List of packets with updated transmission delays and arrival times.
    """

    # List to store received packets
    received = []

    # Initialize last arrival time
    last_arrival_time = Fraction(0, 1)

    # Extract parameters from config
    snr_dB = config['modalities'][modality]['snr_dB']
    bandwidth = config['modalities'][modality]['bandwidth']
    outage_proba = config['modalities'][modality]['outage_proba']

    # Compute the achievable rate
    bandwidth_bps = rate(snr_dB, bandwidth, outage_proba)

    # Convert bandwidth_bps and outage_proba to Fractions if they are floats
    bandwidth_bps_frac = Fraction.from_float(bandwidth_bps).limit_denominator() if isinstance(bandwidth_bps, float) else bandwidth_bps
    outage_proba_frac = Fraction.from_float(outage_proba).limit_denominator() if isinstance(outage_proba, float) else outage_proba

    # Iterate over packets and simulate transmission
    for pkt in stream:
       
        # Compute and store transmission delay as Fraction
        size_bits_frac = Fraction(pkt.size_bits, 1)
        tx_delay = size_bits_frac / bandwidth_bps_frac / (1 - outage_proba_frac)
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