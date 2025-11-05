# src/communication.py

import numpy as np

from typing import List

from src.packets import TransmittedPacket


def rate(epsilon, bandwidth, snr_db):
    """
    Compute the achievable rate using the erasure channel model specified in the paper.

    Args:
        epsilon (float): Outage probability (between 0 and 1).
        bandwidth (float): Channel bandwidth in Hz.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        float: Achievable rate in bits per second.
    """
    snr_linear = 10**(snr_db/10)
    val = 1 - snr_linear * np.log(1 - epsilon)
    val = np.maximum(val, 1e-12)  # prevent log domain error

    return float((bandwidth * np.log2(val)).item())


def simulate_transmission(packets: List[TransmittedPacket], bandwidth_bps, outage_proba):
    """
    Simulate the transmission of packets over a channel with given bandwidth and outage probability. 
    Packets are transmitted sequentially, and their arrival times are computed based on their sizes and the channel conditions.

    Args:
        packets (List[TransmittedPacket]): List of packets to be transmitted.
        bandwidth_bps (float): Channel bandwidth in bits per second.
        outage_proba (float): Outage probability (between 0 and 1).
  
    Returns:
        List[TransmittedPacket]: List of packets with updated transmission delays and arrival times.
    """

    # List to store received packets
    received = []

    # Initialize last arrival time
    last_arrival_time = 0.0

    # Iterate over packets and simulate transmission
    for pkt in packets:
       
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