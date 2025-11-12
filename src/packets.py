# src/packets.py

"""
Packet representation and transmission metadata for media streaming simulation.

This module defines Packet and TransmittedPacket classes encapsulating audio/video media packet
attributes, timestamps, sizes, and transmission-related metadata including delays and loss indicators.

The module also provides a utility function `load_packets` to load and reconstruct transmitted packet lists
from compressed pickled files, associating them with given stream types.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

from typing import List, Optional, Any, Tuple

import gzip
import pickle

from fractions import Fraction


class Packet:
    """
    Unified representation of a generic media packet, audio or video. including transmission metadata.

    Args:
        stream_type (str): Type of the stream, e.g., 'audio' or 'video'.
        pts (Optional[int]): Presentation timestamp as an integer.
        pts_time (Optional[Fraction]): Presentation timestamp as a Fraction in seconds.
        duration (Optional[Fraction]): Duration of the packet in seconds.
        size_bits (int): Size of the packet in bits.
        sample_rate (Optional[float]): Sampling rate if applicable.
        time_base (Optional[Fraction]): Time base denominator of timestamps.
        nb_channels (Optional[int]): Number of audio channels, if audio.
        resolution (Optional[Tuple[int, int]]): Resolution (width, height) for video.
        payload (Any): Payload data of the packet.
        tx_delay (Optional[float]): Transmission delay in seconds (optional).
        arrival_time (Optional[Fraction]): Arrival time fraction (optional).
        is_lost (bool): Flag indicating if lost in transmission (default False).
    """

    def __init__(
        self,
        stream_type: str,
        pts: Optional[int],
        pts_time: Optional[Fraction],
        duration: Optional[Fraction],
        size_bits: int,
        sample_rate: Optional[float],
        time_base: Optional[Fraction],
        nb_channels: Optional[int],
        resolution: Optional[Tuple[int, int]] = None,
        payload: Any = None,
        tx_delay: Optional[float] = None,
        arrival_time: Optional[Fraction] = None,
        is_lost: Optional[bool] = False,
    ) -> None:
        self.stream_type = stream_type
        self.pts = pts
        self.pts_time = pts_time
        self.duration = duration
        self.size_bits = size_bits
        self.sample_rate = sample_rate
        self.time_base = time_base
        self.nb_channels = nb_channels
        self.resolution = resolution
        self.payload = payload
        self.tx_delay = tx_delay
        self.arrival_time = arrival_time
        self.is_lost = is_lost

    def __str__(self) -> str:
        pts_str = str(self.pts) if self.pts is not None else "None"
        pts_time_str = f"{float(self.pts_time):.3f}s" if self.pts_time is not None else "None"
        duration_str = f"{float(self.duration):.3f}s" if self.duration is not None else "None"

        channels_str = str(self.nb_channels) if self.nb_channels is not None else "None"
        resolution_str = f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution else "None"

        tx_delay_str = f"{self.tx_delay:.3f}s" if self.tx_delay is not None else "None"
        arrival_str = f"{float(self.arrival_time):.3f}s" if self.arrival_time is not None else "None"
        lost_str = str(self.is_lost)

        base_str = (f"Packet(type={self.stream_type}, pts={pts_str}, "
                    f"pts_time={pts_time_str}, duration={duration_str}, "
                    f"size={self.size_bits} bits, nb_channels={channels_str}, "
                    f"resolution={resolution_str})")

        transmission_str = (f" tx_delay={tx_delay_str}, arrival_time={arrival_str}, "
                            f"is_lost={lost_str}")

        # Include transmission attributes info only if at least one is present or True
        if any([self.tx_delay is not None, self.arrival_time is not None, self.is_lost]):
            return base_str + "," + transmission_str
        else:
            return base_str

    def __repr__(self) -> str:
        return (f"Packet(stream_type={self.stream_type!r}, pts={self.pts!r}, "
                f"pts_time={self.pts_time!r}, duration={self.duration!r}, "
                f"size_bits={self.size_bits!r}, sample_rate={self.sample_rate!r}, "
                f"time_base={self.time_base!r}, nb_channels={self.nb_channels!r}, "
                f"resolution={self.resolution!r}, payload={self.payload!r}, "
                f"tx_delay={self.tx_delay!r}, arrival_time={self.arrival_time!r}, "
                f"is_lost={self.is_lost!r})")


def load_packets(packets_path: str) -> List[Packet]:
    """
    Load a list of transmitted packets from a compressed pickled file.

    Args:
        packets_path (str): Path to the gzip compressed pickle file with stored packets.

    Returns:
        List[Packet]: List of Packet instances loaded from file.
    """
    # Load packets' data
    with gzip.open(packets_path, 'rb') as f:
        packets = pickle.load(f)

    return packets
