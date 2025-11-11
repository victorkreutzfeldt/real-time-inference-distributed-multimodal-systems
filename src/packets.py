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
    Representation of a generic media packet, audio or video.

    Args:
        stream_type (str): Type of the stream, e.g., 'audio' or 'video'.
        pts (Optional[int]): Presentation timestamp as an integer.
        pts_time (Optional[Fraction]): Presentation timestamp as a Fraction in seconds.
        duration (Optional[Fraction]): Duration of the packet in seconds.
        size_bits (int): Size of the packet in bits.
        sample_rate (Optional[float]): Sampling rate of the audio stream if applicable.
        time_base (Optional[Fraction]): Time base denominator of the timestamps.
        nb_channels (Optional[int]): Number of audio channels, if audio.
        resolution (Optional[Tuple[int, int]]): Resolution (width, height) for video packets.
        payload (Any): Payload data of the packet.
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
        payload: Any = None
    ) -> None:
        self.stream_type: str = stream_type

        self.pts: Optional[int] = pts
        self.pts_time: Optional[float] = pts_time
        self.duration: Optional[float] = duration

        self.size_bits: int = size_bits
        self.sample_rate: Optional[float] = sample_rate
        self.time_base: Optional[Fraction] = time_base
        self.nb_channels: Optional[int] = nb_channels
        self.resolution: Optional[Tuple[int, int]] = resolution

        self.payload = payload


    def __str__(self) -> str:
        pts_str = str(self.pts) if self.pts is not None else "None"
        pts_time_str = f"{self.pts_time:.3f}s" if self.pts_time is not None else "None"
        duration_str = f"{self.duration:.3f}s" if self.duration is not None else "None"
        
        channels_str = str(self.nb_channels) if self.nb_channels is not None else "None"
        resolution_str = f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution else "None"

        return (f"Packet(type={self.stream_type}, pts={pts_str}, "
                f"pts_time={pts_time_str}, duration={duration_str}, "
                f"size={self.size_bits} bits, nb_channels={channels_str}, "
                f"resolution={resolution_str})")


    def __repr__(self) -> str:
        return (f"Packet(stream_type={self.stream_type!r}, pts={self.pts!r}, "
                f"pts_time={self.pts_time!r}, duration={self.duration!r}, "
                f"size_bits={self.size_bits!r}, nb_channels={self.nb_channels!r}, "
                f"resolution={self.resolution!r})")


class TransmittedPacket(Packet):
    """
    Extension of Packet with transmission metadata for simulated network behavior.

    Adds transmission delay, arrival time, and packet loss state.

    Args:
        tx_delay (Optional[float]): Transmission delay in seconds.
        arrival_time (Optional[Fraction]): Arrival time of the packet.
        is_lost (bool): Flag indicating if the packet was lost in transmission.
        kwargs: Arguments to pass to the base Packet class.
    """


    def __init__(
        self,
        tx_delay: Optional[float] = None,
        arrival_time: Optional[Fraction] = None,
        is_lost: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.tx_delay = tx_delay
        self.arrival_time = arrival_time
        self.is_lost = is_lost


    def __str__(self) -> str:
        pts_str = str(self.pts) if self.pts is not None else "None"
        pts_time_str = f"{float(self.pts_time):.3f}s" if self.pts_time is not None else "None"
        tx_delay_str = f"{float(self.tx_delay):.3f}s" if hasattr(self, 'tx_delay') and self.tx_delay is not None else "None"
        arrival_str = f"{float(self.arrival_time):.3f}s" if hasattr(self, 'arrival_time') and self.arrival_time is not None else "None"

        return (f"Packet(pts={pts_str}, pts_time={pts_time_str}, tx_delay={tx_delay_str}, arrival_time={arrival_str})")


    def __repr__(self) -> str:
        return (f"Packet(pts={self.pts!r}, pts_time={self.pts_time!r}, "
                f"tx_delay={getattr(self, 'tx_delay', None)!r}, "
                f"arrival_time={getattr(self, 'arrival_time', None)!r})"
                )


def load_packets(
    stream_type: str,
    packets_path: str

) -> List[TransmittedPacket]:
    """
    Load a list of transmitted packets from a compressed pickled file, reconstructing metadata.

    Args:
        stream_type (str): Stream modality ('audio' or 'video').
        packets_path (str): Path to the gzip compressed pickle file with stored packets.

    Returns:
        List[TransmittedPacket]: List of TransmittedPacket instances loaded from file.
    """
    # Load packets' data
    with gzip.open(packets_path, 'rb') as f:
        packets = pickle.load(f)

    # Prepare to save list of transmitted packets
    transmit_packets = []

    # Go through all packets loaded
    for pkt in packets:
        
        # Create transmit packet
        transmit_pkt = TransmittedPacket(
            stream_type=stream_type,
            pts=pkt.pts,
            pts_time=pkt.pts_time,
            duration=pkt.duration,
            size_bits=pkt.size_bits,
            sample_rate=pkt.sample_rate,
            time_base=pkt.time_base,
            nb_channels=getattr(pkt, 'nb_channels', None),
            resolution=getattr(pkt, 'resolution', None),
            payload=pkt.payload,
            tx_delay=None,
            arrival_time=None,
            is_lost=False  
        )
        transmit_packets.append(transmit_pkt)

    return transmit_packets
