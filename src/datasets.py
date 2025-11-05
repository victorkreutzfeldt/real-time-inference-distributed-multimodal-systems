# src/datasets.py

import ast

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PerVideoMultimodalDataset(Dataset):
    def __init__(self, annotations_file='data/annotations.csv',
                 audio_h5_path='data/classification/embeddings/audio.h5',
                 video_h5_path='data/classification/features/video.h5',
                 split='train',
                 num_tokens=10,
                 num_classes=29,
                 modality='multimodal'):
        """
        Dataset for loading audio and video features per token,
        with multilabels per token.

        Args:
            annotations_file (str): Path to token-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'.
            num_tokens (int): Number of tokens per video.
            num_classes (int): Total number of classes including background.
        """
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


    def __len__(self):
        return len(self.video_ids)


    def __getitem__(self, idx):

        # Get video ID for this index
        video_id = self.video_ids[idx]
        
        sample = {
            'video_id': video_id
        }

        # Prepare lists to hold samples
        if self.modality in ('audio', 'multimodal'):
            audio = torch.tensor(self.audio_h5[video_id][0:self.num_tokens][()], dtype=torch.float32)
            sample['audio'] = audio 

        if self.modality in ('video', 'multimodal'):   
            video = torch.tensor(self.video_h5[video_id][0:self.num_tokens][()], dtype=torch.float32) 
            sample['video'] = video

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
                
        sample['labels'] = token_labels  # (num_tokens, num_classes)

        return sample

    def close(self):
        self.audio_h5.close()
        self.video_h5.close()


class PerVideoMultimodalDatasetLabels(Dataset):
    def __init__(self, annotations_file='data/annotations.csv',
                 split='train',
                 num_tokens=10,
                 num_classes=29):
        """
        Dataset for loading labels token-wise only.

        Args:
            annotations_file (str): Path to token-level annotation CSV.
            split (str): 'train', 'val', or 'test'.
            num_tokens (int): Number of tokens per video.
            num_classes (int): Total number of classes including background.
        """
        self.num_tokens = num_tokens
        self.num_classes = num_classes

        # Load and filter by split
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        # Unique video IDs in this split
        self.video_ids = self.annotations['video_id'].unique()

        # Load label mapping from precomputed label_idx column
        self.label_column = 'labels_idx'

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):

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


