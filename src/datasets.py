# src/datasets.py

"""
This module defines custom PyTorch Dataset classes for loading multimodal video data.

Classes:
    - PerVideoMultimodalDataset: Loads tokenized audio and video embeddings along with multilabel annotations.
    - PerVideoMultimodalDatasetLabels: Loads only token-wise multilabel annotations without any embeddings.

These datasets handle reading data from HDF5 files and CSV annotations, supporting flexible
modalities ('audio', 'video', 'multimodal') and providing easy-to-use interfaces for training
or evaluating multimodal video models.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import ast

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PerVideoMultimodalDataset(Dataset):
    """
    Dataset that loads per-token audio and video embeddings along with multilabel annotations.

    Supports modalities 'audio', 'video', or 'multimodal' and loads data from HDF5 files and CSV annotations.

    Args:
        annotations_file (str): Path to the CSV file containing token-level annotations.
        audio_h5_path (str): Filepath for audio embeddings stored in HDF5 format.
        video_h5_path (str): Filepath for video embeddings stored in HDF5 format.
        split (str): Data split to load ('train', 'val', 'test').
        num_tokens (int): Number of tokens per video input.
        num_classes (int): Total number of label classes including background.
        modality (str): One of 'audio', 'video', 'multimodal' specifying modalities to load.
    """
    def __init__(self, 
                 annotations_file: str = 'data/annotations.csv',
                 audio_h5_path: str = 'data/classification/embeddings/audio.h5',
                 video_h5_path: str = 'data/classification/features/video.h5',
                 split: str = 'train',
                 num_tokens: int = 10,
                 num_classes: int = 29,
                 modality: str = 'multimodal') -> None:
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.modality = modality

        # Load and filter annotations by split
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)
        self.label_column = 'labels_idx'

        # Open HDF5 files depending on modality
        self.audio_h5 = None
        self.video_h5 = None
        
        if self.modality in ('audio', 'multimodal'):
            self.audio_h5 = h5py.File(audio_h5_path, 'r')

        if self.modality in ('video', 'multimodal'):
            self.video_h5 = h5py.File(video_h5_path, 'r')

        # Unique video IDs in this split
        self.video_ids = self.annotations['video_id'].unique()
        
    def __len__(self) -> int:
        """
        Returns:
            int: Number of unique videos in the selected split.
        """
        return len(self.video_ids)

    def __getitem__(self, idx) -> dict:
        """
        Retrieve modalities and multilabel token annotations for a single video by index.

        Args:
            idx (int): Index of the video input.

        Returns:
            dict: Input dictionary with keys:
                - 'video_id' (str): Video identifier.
                - 'audio' (torch.FloatTensor): Audio embeddings if modality includes 'audio', shape (num_tokens, feature_dim).
                - 'video' (torch.FloatTensor): Video embeddings if modality includes 'video', shape (num_tokens, feature_dim).
                - 'labels' (torch.FloatTensor): Multi-hot encoded labels per token, shape (num_tokens, num_classes).

        Raises:
            AssertionError: If annotation tokens count does not match 'num_tokens'.
        """
        # Get video ID for this index
        video_id = self.video_ids[idx]
        
        # Prepare to output an input
        input = {
            'video_id': video_id
        }

        # Prepare lists to hold inputs
        if self.modality in ('audio', 'multimodal'):
            audio = torch.tensor(self.audio_h5[video_id][0:self.num_tokens][()], dtype=torch.float32)
            input['audio'] = audio 

        if self.modality in ('video', 'multimodal'):   
            video = torch.tensor(self.video_h5[video_id][0:self.num_tokens][()], dtype=torch.float32) 
            input['video'] = video

        # Filter annotations for this video
        video_anns = self.annotations[self.annotations['video_id'] == video_id][self.label_column].reset_index(drop=True)
        
        # Sanity check
        assert len(video_anns) == self.num_tokens, \
            f"Expected {self.num_tokens} tokens for video {video_id}, found {len(video_anns)}."  

        # Per-token multilabels: shape (num_tokens, num_classes)
        token_labels = torch.zeros((self.num_tokens, self.num_classes), dtype=torch.float32)

        # Iterate through tokens
        for token_idx in range(self.num_tokens):
            
            # Load labels for this token, mark multilabel vector
            labels_this_token_list = ast.literal_eval(video_anns[token_idx])

            # Go through each label index and one-hot encode
            for lbl_idx in labels_this_token_list:
                token_labels[token_idx, lbl_idx] = 1.0
                
        input['labels'] = token_labels  # (num_tokens, num_classes)

        return input
    
    def close(self) -> None:
        """
        Close any open HDF5 file handles to release resources.
        """
        self.audio_h5.close()
        self.video_h5.close()


class PerVideoMultimodalDatasetLabels(Dataset):
    """
    Dataset class for loading token-wise multi-label annotations without embeddings.

    Useful for scenarios where only ground-truth labels are required.

    Args:
        annotations_file (str): Path to CSV with token-level annotations.
        split (str): Desired data split - 'train', 'val', 'test'.
        num_tokens (int): Number of tokens per video input.
        num_classes (int): Number of label classes including background.
    """
    def __init__(self, 
                 annotations_file='data/annotations.csv',
                 split='train',
                 num_tokens=10,
                 num_classes=29) -> None:

        self.num_tokens = num_tokens
        self.num_classes = num_classes

        # Load and filter by split
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        # Unique video IDs in this split
        self.video_ids = self.annotations['video_id'].unique()

        # Load label mapping from precomputed label_idx column
        self.label_column = 'labels_idx'

    def __len__(self) -> int:
        """
        Returns:
            int: Number of unique videos in the selected split.
        """
        return len(self.video_ids)

    def __getitem__(self, idx) -> dict:
        """
        Retrieve multi-hot label annotations for all tokens of a video.

        Args:
            idx (int): Index of the video input.

        Returns:
            dict: Input dictionary containing:
                - 'video_id' (str): Video identifier.
                - 'labels' (torch.FloatTensor): Multi-hot label tensor per token, shape (num_tokens, num_classes).

        Raises:
            AssertionError: If annotation tokens count does not match `num_tokens`.
        """

        # Get video ID for this index
        video_id = self.video_ids[idx]

        # Filter annotations for this video
        video_anns = self.annotations[self.annotations['video_id'] == video_id][self.label_column].reset_index(drop=True)
        
        # Sanity check
        assert len(video_anns) == self.num_tokens, \
            f"Expected {self.num_tokens} tokens for video {video_id}, found {len(video_anns)}."  

        # Per-token multilabels: shape (num_tokens, num_classes)
        token_labels = torch.zeros((self.num_tokens, self.num_classes), dtype=torch.float32)

        # Iterate through tokens
        for token_idx in range(self.num_tokens):
            
            # Load labels for this token, mark multilabel vector
            labels_this_token_list = ast.literal_eval(video_anns[token_idx])

            # Go through each label index and one-hot encode
            for lbl_idx in labels_this_token_list:
                token_labels[token_idx, lbl_idx] = 1.0

        return {
            'video_id': video_id,
            'labels': token_labels
        }