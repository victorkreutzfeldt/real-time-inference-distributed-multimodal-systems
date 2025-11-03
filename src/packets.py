from typing import List, Optional, Any, Tuple

import gzip
import pickle

import h5py
import numpy as np

from fractions import Fraction

class Packet:
    """
    Packet class representing a generic media packet (audio or video). 
    """
    def __init__(self,
                stream_type: str,
                pts: Optional[int],
                pts_time: Optional[Fraction],
                duration: Optional[Fraction],
                size_bits: int,
                sample_rate: Optional[float],
                time_base: Optional[Fraction],
                nb_channels: Optional[int],
                resolution: Optional[Tuple[int, int]] = None
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
    TransmittedPacket class representing a media packet with transmission metadata.
    """
    def __init__(self,
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
                 is_lost: bool = False,
                 pts_time_drifted: Optional[Fraction] = None
                 ) -> None:
        
        super().__init__(stream_type, pts, pts_time, duration, size_bits, sample_rate, time_base , nb_channels, resolution)

        self.payload = payload
        self.tx_delay = tx_delay
        self.arrival_time = arrival_time
        self.is_lost = is_lost
        self.pts_time_drifted = pts_time_drifted

    def __str__(self) -> str:
        pts_str = str(self.pts) if self.pts is not None else "None"
        pts_time_str = f"{float(self.pts_time):.3f}s" if self.pts_time is not None else "None"
        tx_delay_str = f"{float(self.tx_delay):.3f}s" if hasattr(self, 'tx_delay') and self.tx_delay is not None else "None"
        arrival_str = f"{float(self.arrival_time):.3f}s" if hasattr(self, 'arrival_time') and self.arrival_time is not None else "None"
        drift_str = f"{float(self.pts_time_drifted):.3f}s" if hasattr(self, 'pts_time_drifted') and self.pts_time_drifted is not None else "None"

        return (f"Packet(pts={pts_str}, pts_time={pts_time_str}, tx_delay={tx_delay_str}, "
            f"arrival_time={arrival_str}, pts_time_drifted={drift_str})")

    def __repr__(self) -> str:
        return (f"Packet(pts={self.pts!r}, pts_time={self.pts_time!r}, "
                f"tx_delay={getattr(self, 'tx_delay', None)!r}, "
                f"arrival_time={getattr(self, 'arrival_time', None)!r}, "
                f"pts_time_drifted={getattr(self, 'pts_time_drifted', None)!r})")


def load_transmitted_packets_from_saved_features(
    packets_path: str,
    payload_path: str,
    pts2idx_path: str,
    stream_type: str
) -> List[TransmittedPacket]:
    """
    Loads pre-extracted packets (from HDF5 + pickled metadata) and attaches decoded payloads.
    """

    # Load packets' data
    with gzip.open(packets_path, 'rb') as f:
        packets = pickle.load(f)

    with h5py.File(payload_path, 'r') as h5f:
        payloads = h5f['payloads'][:]

    with gzip.open(pts2idx_path, 'rb') as f:
        pts2idx = pickle.load(f)

    # Prepare to save list of transmitted packets
    transmit_packets = []

    # Go through all packets loaded
    for pkt in packets:

        # Get the packet index
        idx = pts2idx[pkt.pts]
    
        # Load the payload
        payload = payloads[idx]
        
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
            payload=payload,
            tx_delay=None,
            arrival_time=None,
            is_lost=False,
            pts_time_drifted=None    
        )
        transmit_packets.append(transmit_pkt)

    return transmit_packets
