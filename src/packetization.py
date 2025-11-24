# src/packetization.py

"""
Packet representation and transmission metadata for media streaming simulation.

This module defines the `Packet` class, representing audio and video media packets with
associated attributes such as timestamps, size, payload, and transmission metadata including
delay and loss information.

It includes the `load_packets` utility to load transmitted packet lists from compressed
pickle files.

The `Packetization` class provides functionality to extract raw media data using FFmpeg,
packetize audio and video streams into `Packet` instances, save packets to disk, and clean
temporary directories.

Author: Victor Kreützfeldt (@victorkreutzfeldt)
Date: 2025-11-11
"""

import os

import subprocess
from fractions import Fraction
from typing import List, Tuple, Optional, Any

import numpy as np
from PIL import Image
import gzip
import pickle
from scipy.io import wavfile


class Packet:
    """
    Unified representation of a generic media packet, audio or video. including transmission metadata.

    Args:
        stream_type (str): Type of the stream, e.g., 'audio' or 'video'.
        pts (Optional[int]): Presentation timestamp as an integer.
        pts_time (Optional[Fraction]): Presentation timestamp as a Fraction in seconds.
        duration (Optional[Fraction]): Duration of the packet in seconds.
        size_bits (int): Size of the packet in bits.
        sampling_rate (Optional[float]): Sampling rate if applicable.
        time_base (Optional[Fraction]): Time base denominator of timestamps.
        num_channels (Optional[int]): Number of audio channels, if audio.
        resolution (Optional[Tuple[int, int]]): Resolution (width, height) for video.
        payload (Any): Payload data of the packet.
        tx_delay (Optional[Faction]): Transmission delay in seconds (optional).
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
        sampling_rate: Optional[float],
        time_base: Optional[Fraction],
        num_channels: Optional[int],
        resolution: Optional[Tuple[int, int]] = None,
        payload: Any = None,
        tx_delay: Optional[Fraction] = None,
        arrival_time: Optional[Fraction] = None,
        is_lost: Optional[bool] = False,
    ) -> None:
        self.stream_type = stream_type
        self.pts = pts
        self.pts_time = pts_time
        self.duration = duration
        self.size_bits = size_bits
        self.sampling_rate = sampling_rate
        self.time_base = time_base
        self.num_channels = num_channels
        self.resolution = resolution
        self.payload = payload
        self.tx_delay = tx_delay
        self.arrival_time = arrival_time
        self.is_lost = is_lost

    def __str__(self) -> str:
        pts_str = str(self.pts) if self.pts is not None else "None"
        pts_time_str = f"{float(self.pts_time):.3f}s" if self.pts_time is not None else "None"
        duration_str = f"{float(self.duration):.3f}s" if self.duration is not None else "None"

        channels_str = str(self.num_channels) if self.num_channels is not None else "None"
        resolution_str = f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution else "None"

        tx_delay_str = f"{self.tx_delay:.3f}s" if self.tx_delay is not None else "None"
        arrival_str = f"{float(self.arrival_time):.3f}s" if self.arrival_time is not None else "None"
        lost_str = str(self.is_lost)

        base_str = (f"Packet(type={self.stream_type}, pts={pts_str}, "
                    f"pts_time={pts_time_str}, duration={duration_str}, "
                    f"size={self.size_bits} bits, num_channels={channels_str}, "
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
                f"size_bits={self.size_bits!r}, sampling_rate={self.sampling_rate!r}, "
                f"time_base={self.time_base!r}, num_channels={self.num_channels!r}, "
                f"resolution={self.resolution!r}, "
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


class Packetization:
    """
    Handles extraction, packetization, serialization, and cleanup of audio and video
    streams for simulation of media transmission.

    Attributes:
        config (dict): Configuration dictionary defining modalities, paths, sampling rates,
                       resolutions, and other packetization parameters.
    """
    def __init__(self, config):
        self.config = config

        # Check if folders exist
        for m in config['modalities'].keys():
            os.makedirs(config['modalities'][m]['tmp_dir'], exist_ok=True)
            os.makedirs(config['modalities'][m]['packets_dir'], exist_ok=True) 

    def packetize_single_video(self, vid, video_payloads) -> dict:
        """
        Packetize a single video stream across all modalities.

        Args:
            vid (str): Video identifier.
            video_payloads (Optional[np.ndarray]): Video feature representations to packetize.

        Returns:
            dict: Output dictionary containing counts of generated packets per modality.
        """
        # Initialize output dictionary with packet_counts as an empty dict
        out = {'packet_counts': {}}

        # Go over modalities
        for m in self.config['modalities'].keys():

            # Create entry if it does not exist
            if m not in out['packet_counts']:
                out['packet_counts'][m] = None

            # Extract raw data
            self.extract_raw_data_ffmpeg(vid, modality=m)

            # Packetize raw data
            packet_list = self.packetize(video_payloads, modality=m)

            # Save list of packets
            self.save_packet_list(vid, packet_list, modality=m)

            # Update output
            out['packet_counts'][m] = len(packet_list)

        # Clean up temporary folders
        self.cleanup_temp_files()

        return out

    def extract_raw_data_ffmpeg(self, vid, modality) -> None:
        """
        Extract raw audio or video data from input video file using FFmpeg.

        Args:
            vid (str): Video identifier.
            modality (str): Modality key ('audio' or 'video').
        """
        # Get video path
        video_path = os.path.join(self.config['global']['videos_dir'], f"{vid}.avi")

        # Get parameters
        sampling_rate = self.config['modalities'][modality]['sampling_rate']
        resolution = self.config['modalities'][modality]['resolution'] 

        # Get temporary dir 
        tmp_dir = self.config['modalities'][modality]['tmp_dir']

        # Prepare FFmpeg command according to modality
        if modality == 'audio':

            cmd = [
                'ffmpeg', '-y', '-loglevel', 'quiet',
                '-i', video_path,
                '-ac', '1',                 # Force mono audio channel
                '-ar', str(sampling_rate),  # Set audio sampling rate
                '-vn',                      # Disable video processing
                os.path.join(tmp_dir, 'waveform.wav')
            ]
        
        elif modality == 'video':
    
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'quiet',
                '-i', video_path,
                '-vf', 
                f'fps={sampling_rate},scale={resolution[0]}:{resolution[1]}',
                os.path.join(tmp_dir, 'frame_%05d.png')
            ]

        # Run command
        subprocess.run(cmd, check=True)

    def packetize(self, video_payloads, modality) -> List[Packet]:
        """
        Create packets from raw extracted data or video representations for a given modality.

        Args:
            video_payloads (Optional[np.ndarray]): Precomputed video feature payloads.
            modality (str): Modality key specifying 'audio' or 'video'.

        Returns:
            List[Packet]: A list of Packet objects for the given modality.
        """
        # Get temporary dir 
        tmp_dir = self.config['modalities'][modality]['tmp_dir']

        # Get parameters
        sampling_rate = self.config['modalities'][modality]['sampling_rate']
        num_channels = self.config['modalities'][modality]['num_channels']
        resolution = self.config['modalities'][modality]['resolution']
        packet_size_samples = self.config['modalities'][modality]['packet_size_samples'] 

        # Prepare to save packets using the Packet class
        packet_list = []

        # Calculate the packet duration
        packet_duration = Fraction(packet_size_samples, sampling_rate)

        if modality == 'audio':
            
            # Get file
            file = next((f for f in os.listdir(tmp_dir) if f.endswith('.wav')), None)
            
            if file is None:
                raise FileNotFoundError("No .wav file found in the directory.")
            
            # Extract waveform
            sr, waveform = wavfile.read(os.path.join(tmp_dir, file))

            # Sanity check
            assert sr == sampling_rate, f"Expected {sampling_rate}, got {sr}."
            assert waveform.ndim == num_channels, f"Expected mono audio, got {waveform.ndim} channels."

            # Normalize waveform
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0

            # Get expected number of samples
            target_num_samples = int(self.config['global']['video_duration'] * sampling_rate)

            # Make waveform to be compliant with what is expected
            if len(waveform) < target_num_samples:
                waveform = np.pad(waveform, (0, target_num_samples - len(waveform)), mode='constant')
            else:
                waveform = waveform[:target_num_samples]

            # Packetize waveform given the defined packet size in samples
            packets = waveform.reshape((packet_size_samples, -1)).T

            # Calculate the packet size in bits
            packet_size_bits = packet_size_samples * 16 # PCM-16: 16 bits per sample

            # Go over packets 
            for pp, packet in enumerate(packets):
                pkt = Packet(
                    stream_type='audio',
                    pts=pp,
                    pts_time=packet_duration * pp,
                    duration=packet_duration,
                    size_bits=packet_size_bits,
                    sampling_rate=sampling_rate,
                    time_base=Fraction(1, sampling_rate),
                    num_channels=num_channels,
                    payload=packet.astype(np.float32)
                )
                packet_list.append(pkt)

            return packet_list

        elif modality == 'video':

            # Get files
            files = sorted(f for f in os.listdir(tmp_dir) if f.endswith('.png'))

            # Get expected number of samples
            # For video, each image or sample is considered to be a packet
            target_num_samples = int(self.config['global']['video_duration'] * sampling_rate)

            if len(files) > target_num_samples:
                files = files[:target_num_samples]
            else:
                pass # TODO: handles this case, not found for this dataset 

            # Sanity check
            assert len(files) == target_num_samples, f"Expected {target_num_samples}, got {len(files)}"

            if video_payloads is None:
                raise RuntimeError(f"Video payloads are not available.")
            
            # Reshape video payloads
            video_payloads = video_payloads.reshape((-1, 512, 7, 7))

            # Go over packets
            for pp, fname in enumerate(files):
                if pp == 0:
                    path = os.path.join(tmp_dir, fname)
                    img = Image.open(path).convert('RGB')
                    np_img = np.array(img, dtype=np.uint8)
                    packet_size_bits = np_img.nbytes * 8

                pkt = Packet(
                    stream_type='video',
                    pts=pp,
                    pts_time=packet_duration * pp,
                    duration=packet_duration,
                    size_bits=packet_size_bits,
                    sampling_rate=sampling_rate,
                    time_base=Fraction(1, sampling_rate),
                    num_channels=3,
                    resolution=resolution[::-1],
                    payload=video_payloads[pp] 
                )
                packet_list.append(pkt)

            return packet_list
    
    def save_packet_list(self, vid, packet_list, modality) -> None:
        """
        Serialize and save a list of packets to a compressed pickle file.

        Args:
            vid (str): Video identifier.
            packet_list (List[Packet]): List of Packet instances to save.
            modality (str): Modality key.
        """
        packet_dir = self.config['modalities'][modality]['packets_dir']
        packets_path = os.path.join(packet_dir, f'{vid}.pkl.gz')

        with gzip.open(packets_path, 'wb') as f:
                pickle.dump(packet_list, f)

    def cleanup_temp_files(self) -> None: 
        """
        Remove all files within temporary directories for all modalities without deleting the directories.
        """
        # Iterate over all modalities
        for m in self.config['modalities'].keys():
            tmp_dir = self.config['modalities'][m]['tmp_dir']

            # List all files in the tmp_dir
            for filename in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, filename)

                # Check if it's a file before deleting (to avoid deleting subdirectories)
                if os.path.isfile(file_path):
                    os.remove(file_path)