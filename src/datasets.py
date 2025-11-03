import ast
import os

import h5py
import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import itertools

def one_hot_encode(labels, num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[labels] = 1.0
    return vec


class PerChunkMultimodalDataset(Dataset):
    def __init__(self, 
                 annotations_file='data/annotations.csv',
                 audio_h5_path='data/classification/embeddings/audio.h5',
                 video_h5_path='data/classification/features/video.h5',
                 split='train',
                 num_classes=29,
                 modality='multimodal'):
        """
        Dataset for loading chunk-level features and labels.

        Args:
            annotations_file (str): Path to the chunk-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'
            num_classes (int): Number of label classes for one-hot encoding.
            modality (str): 'audio', 'video', or 'multimodal'
        """
        super().__init__()

        assert modality in ('audio', 'video', 'multimodal'), \
            f"Invalid modality '{modality}', choose from 'audio', 'video', 'multimodal'."
        self.modality = modality
        self.num_classes = num_classes

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

        # TODO: Create index vector

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_id = row['video_id']
        chunk_idx = int(row['chunk_idx']) 

        sample = {
            'video_id': video_id,
            'chunk_idx': chunk_idx
        }

        # Conditionally add features based on modality
        if self.modality in ('audio', 'multimodal'):
            audio = torch.tensor(self.audio_h5[video_id][chunk_idx], dtype=torch.float32)
            sample['audio'] = audio

        if self.modality in ('video', 'multimodal'):
            video = torch.tensor(self.video_h5[video_id][chunk_idx], dtype=torch.float32)
            sample['video'] = video

        # Label processing
        label_list = ast.literal_eval(row[self.label_column])
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        labels[label_list] = 1.0
        sample['labels'] = labels

        return sample

    def close(self):
        """Close any open HDF5 file handles."""
        if self.audio_h5 is not None:
            self.audio_h5.close()
        if self.video_h5 is not None:
            self.video_h5.close()


# class PerTimeStepAlignDataset(Dataset):
#     def __init__(self, 
#                  annotations_file='data/annotations.csv',
#                  audio_h5_path='data/audio_features.h5',
#                  video_h5_path='data/video_features.h5',
#                  split='train',
#                  modality='multimodal',
#                  num_chunks=10,
#                  num_timesteps=96
#                  ):
#         """
#         Dataset for loading chunk-level features and labels.

#         Args:
#             annotations_file (str): Path to the chunk-level annotation CSV.
#             audio_h5_path (str): HDF5 file with audio features.
#             video_h5_path (str): HDF5 file with video features.
#             split (str): 'train', 'val', or 'test'
#             num_classes (int): Number of label classes for one-hot encoding.
#             modality (str): 'audio', 'video', or 'multimodal'
#         """
#         super().__init__()

#         assert modality in ('audio', 'video', 'multimodal'), \
#             f"Invalid modality '{modality}', choose from 'audio', 'video', 'multimodal'."
#         self.modality = modality

#         # Load and filter by split
#         self.annotations = pd.read_csv(annotations_file)
#         self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

#         # Unique video IDs in this split
#         self.video_ids = self.annotations['video_id'].unique()

#         # Open HDF5 files depending on modality
#         self.audio_h5 = None
#         self.video_h5 = None

#         if self.modality in ('audio', 'multimodal'):
#             self.audio_h5 = h5py.File(audio_h5_path, 'r')
        
#         if self.modality in ('video', 'multimodal'):
#             self.video_h5 = h5py.File(video_h5_path, 'r') 
#             self.video_transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.422, 0.392, 0.367],
#                     std=[0.277, 0.275, 0.280])
#                 ])

#         # Number of chunks per video
#         self.num_chunks = num_chunks

#         # Number of time steps
#         self.num_timesteps = num_timesteps

#         # Create index vector
#         self.index_vector = []
#         for v_idx, video_id in enumerate(self.video_ids):
#             for c_idx in range(self.num_chunks):
#                 for t_idx in range(self.num_timesteps):
#                     self.index_vector.append((v_idx, c_idx, t_idx))
                    
#     def __len__(self):
#         return len(self.index_vector)

#     def __getitem__(self, idx): 
#         v_idx, c_idx, t_idx = self.index_vector[idx]
#         video_id = self.video_ids[v_idx]
#         sample = {'video_id': video_id, 'chunk': c_idx, 'timestep': t_idx}

#         if self.modality in ('audio', 'multimodal'):
#             # For time step: get [chunk][timestep]
#             audio = self.audio_h5[video_id][c_idx][t_idx]
#             sample['audio'] = torch.tensor(audio, dtype=torch.float32)

#         if self.modality in ('video', 'multimodal'):
#             img = self.video_h5[video_id][c_idx][t_idx]  # single image (H,W,C)
#             img = self.video_transform(img)
#             sample['video'] = img

#         return sample

#     def close(self):
#         """Close any open HDF5 file handles."""
#         if self.audio_h5 is not None:
#             self.audio_h5.close()
#         if self.video_h5 is not None:
#             self.video_h5.close()


class PerVideoMultimodalDataset(Dataset):
    def __init__(self, annotations_file='data/annotations.csv',
                 audio_h5_path='data/classification/embeddings/audio.h5',
                 video_h5_path='data/classification/features/video.h5',
                 split='train',
                 num_chunks=10,
                 num_classes=29,
                 modality='multimodal'):
        """
        Dataset for loading audio and video features per chunk,
        with multilabels per chunk.

        Args:
            annotations_file (str): Path to chunk-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'.
            num_chunks (int): Number of chunks per video.
            num_classes (int): Total number of classes including background.
        """
        self.num_chunks = num_chunks
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
            audio = torch.tensor(self.audio_h5[video_id][0:self.num_chunks][()], dtype=torch.float32)
            sample['audio'] = audio 

        if self.modality in ('video', 'multimodal'):   
            video = torch.tensor(self.video_h5[video_id][0:self.num_chunks][()], dtype=torch.float32) 
            sample['video'] = video

        # Filter annotations for this video
        video_anns = self.annotations[self.annotations['video_id'] == video_id][self.label_column].reset_index(drop=True)
        
        # Sanity check
        assert len(video_anns) == self.num_chunks, \
            f"Expected {self.num_chunks} chunks for video {video_id}, found {len(video_anns)}."  

        # Per-chunk multilabels: shape (num_chunks, num_classes)
        chunk_labels = torch.zeros((self.num_chunks, self.num_classes), dtype=torch.float32)

        # Iterate through chunks
        for chunk_idx in range(self.num_chunks):
            
            # Load labels for this chunk, mark multilabel vector
            labels_this_chunk_list = ast.literal_eval(video_anns[chunk_idx])

            # Go through each label index and one-hot encode
            for lbl_idx in labels_this_chunk_list:
                chunk_labels[chunk_idx, lbl_idx] = 1.0
                
        sample['labels'] = chunk_labels  # (num_chunks, num_classes)

        return sample

    def close(self):
        self.audio_h5.close()
        self.video_h5.close()


class PerVideoMultimodalDatasetLabels(Dataset):
    def __init__(self, annotations_file='data/annotations.csv',
                 split='train',
                 num_chunks=10,
                 num_classes=29):
        """
        Dataset for loading audio and video features per chunk,
        with multilabels per chunk.

        Args:
            annotations_file (str): Path to chunk-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'.
            num_chunks (int): Number of chunks per video.
            num_classes (int): Total number of classes including background.
        """
        self.num_chunks = num_chunks
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
        assert len(video_anns) == self.num_chunks, \
            f"Expected {self.num_chunks} chunks for video {video_id}, found {len(video_anns)}."  

        # Per-chunk multilabels: shape (num_chunks, num_classes)
        chunk_labels = torch.zeros((self.num_chunks, self.num_classes), dtype=torch.float32)

        # Iterate through chunks
        for chunk_idx in range(self.num_chunks):
            
            # Load labels for this chunk, mark multilabel vector
            labels_this_chunk_list = ast.literal_eval(video_anns[chunk_idx])

            # Go through each label index and one-hot encode
            for lbl_idx in labels_this_chunk_list:
                chunk_labels[chunk_idx, lbl_idx] = 1.0

        return {
            'video_id': video_id,
            'labels': chunk_labels
        }
    

class PerChunkAlignDataset(Dataset):
    def __init__(self, 
                 annotations_file='data/annotations.csv',
                 audio_h5_path='data/audio_features.h5',
                 video_h5_path='data/video_features.h5',
                 split='train',
                 modality='multimodal',
                 num_chunks=10
                 ):
        """
        Dataset for loading chunk-level features and labels.

        Args:
            annotations_file (str): Path to the chunk-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'
            num_classes (int): Number of label classes for one-hot encoding.
            modality (str): 'audio', 'video', or 'multimodal'
        """
        super().__init__()

        assert modality in ('audio', 'video', 'multimodal'), \
            f"Invalid modality '{modality}', choose from 'audio', 'video', 'multimodal'."
        self.modality = modality

        # Load and filter by split
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        # Unique video IDs in this split
        self.video_ids = self.annotations['video_id'].unique()

        # Open HDF5 files depending on modality
        self.audio_h5 = None
        self.video_h5 = None

        if self.modality in ('audio', 'multimodal'):
            self.audio_h5 = h5py.File(audio_h5_path, 'r')
        
        if self.modality in ('video', 'multimodal'):
            self.video_h5 = h5py.File(video_h5_path, 'r') 

        # Number of chunks per video
        self.num_chunks = num_chunks

        # Create index vector
        self.index_vector = []
        for v_idx, video_id in enumerate(self.video_ids):
            for c_idx in range(self.num_chunks):
                self.index_vector.append((v_idx, c_idx))
                    
    def __len__(self):
        return len(self.index_vector)

    def __getitem__(self, idx): 
        v_idx, c_idx = self.index_vector[idx]
        video_id = self.video_ids[v_idx]
        sample = {'video_id': video_id, 'chunk': c_idx}

        if self.modality in ('audio', 'multimodal'):
            # For time step: get [chunk][timestep]
            audio = self.audio_h5[video_id][c_idx]
            sample['audio'] = torch.tensor(audio, dtype=torch.float32)

        if self.modality in ('video', 'multimodal'):
            video = self.video_h5[video_id][c_idx]
            sample['video'] = video

        return sample

    def close(self):
        """Close any open HDF5 file handles."""
        if self.audio_h5 is not None:
            self.audio_h5.close()
        if self.video_h5 is not None:
            self.video_h5.close()


class PerTimeStepAlignDataset(Dataset):
    def __init__(self, 
                 annotations_file='data/annotations.csv',
                 audio_h5_path='data/audio_features.h5',
                 video_h5_path='data/video_features.h5',
                 split='train',
                 modality='multimodal',
                 num_chunks=10,
                 num_timesteps=6
                 ):
        """
        Dataset for loading chunk-level features and labels.

        Args:
            annotations_file (str): Path to the chunk-level annotation CSV.
            audio_h5_path (str): HDF5 file with audio features.
            video_h5_path (str): HDF5 file with video features.
            split (str): 'train', 'val', or 'test'
            num_classes (int): Number of label classes for one-hot encoding.
            modality (str): 'audio', 'video', or 'multimodal'
        """
        super().__init__()

        assert modality in ('audio', 'video', 'multimodal'), \
            f"Invalid modality '{modality}', choose from 'audio', 'video', 'multimodal'."
        self.modality = modality

        # Load and filter by split
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        # Unique video IDs in this split
        self.video_ids = self.annotations['video_id'].unique()

        # Open HDF5 files depending on modality
        self.audio_h5 = None
        self.video_h5 = None

        if self.modality in ('audio', 'multimodal'):
            self.audio_h5 = h5py.File(audio_h5_path, 'r')
        
        if self.modality in ('video', 'multimodal'):
            self.video_h5 = h5py.File(video_h5_path, 'r') 

        # Number of chunks per video
        self.num_chunks = num_chunks

        # Number of time steps
        self.num_timesteps = num_timesteps

        # Create index vector
        self.index_vector = []
        for v_idx, video_id in enumerate(self.video_ids):
            for c_idx in range(self.num_chunks):
                for t_idx in range(self.num_timesteps):
                    self.index_vector.append((v_idx, c_idx, t_idx))
                    
    def __len__(self):
        return len(self.index_vector)

    def __getitem__(self, idx): 
        v_idx, c_idx, t_idx = self.index_vector[idx]
        video_id = self.video_ids[v_idx]
        sample = {'video_id': video_id, 'chunk': c_idx, 'timestep': t_idx}

        if self.modality in ('audio', 'multimodal'):
            # For time step: get [chunk][timestep]
            audio = self.audio_h5[video_id][c_idx][t_idx]
            sample['audio'] = torch.tensor(audio, dtype=torch.float32)

        if self.modality in ('video', 'multimodal'):
            video = self.video_h5[video_id][c_idx][t_idx]
            sample['video'] = video

        return sample

    def close(self):
        """Close any open HDF5 file handles."""
        if self.audio_h5 is not None:
            self.audio_h5.close()
        if self.video_h5 is not None:
            self.video_h5.close()




# class PerVideoExamplesDataset(Dataset):
#     def __init__(self, annotations_file='data/annotations.csv',
#                  audio_h5_path=None,
#                  video_h5_path=None,
#                  split='train',
#                  num_chunks=10,
#                  num_classes=29):
#         """
#         Dataset for loading audio and video features per chunk,
#         with multilabels per chunk.

#         Args:
#             annotations_file (str): Path to chunk-level annotation CSV.
#             audio_h5_path (str): HDF5 file with audio features.
#             video_h5_path (str): HDF5 file with video features.
#             split (str): 'train', 'val', or 'test'.
#             num_chunks (int): Number of chunks per video.
#             num_classes (int): Total number of classes including background.
#         """
#         self.num_chunks = num_chunks
#         self.num_classes = num_classes

#         # Load and filter by split
#         self.annotations = pd.read_csv(annotations_file)
#         self.annotations = self.annotations[self.annotations['split'] == split]

#         # Unique video IDs in this split
#         self.video_ids = self.annotations['video_id'].unique()

#         # Load label mapping from precomputed label_idx column
#         self.label_column = 'labels_idx'
       
#         # Open HDF5 files
#         self.audio_h5 = h5py.File(audio_h5_path, 'r')
#         self.video_h5 = h5py.File(video_h5_path, 'r')

#         # Map label names to indices
#         self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}

#     def __len__(self):
#         return len(self.video_ids)

#     def __getitem__(self, idx):

#         # Get video ID for this index
#         video_id = self.video_ids[idx]

#         # Initialize lists to hold features
#         video_feats = []
#         audio_feats = []

#         # Per-chunk multilabels: shape (num_chunks, num_classes)
#         chunk_labels = np.zeros((self.num_chunks, self.num_classes), dtype=np.float32)
    
#         # Filter annotations for this video
#         video_anns = self.annotations[self.annotations['video_id'] == video_id]

#         for chunk_idx in range(self.num_chunks):
            
#             # Go through each row in the annotations for this video
#             row = video_anns.iloc[chunk_idx]

#             # Load features from HDF5 using string keys
#             video_chunk_feat = self.video_h5[video_id][chunk_idx][()]  # e.g. (16, 512, 7, 7)
#             audio_chunk_feat = self.audio_h5[video_id][chunk_idx][()]  # e.g. (1, 96, 64)
    
#             video_feats.append(video_chunk_feat)
#             audio_feats.append(audio_chunk_feat)

#             # Load labels for this chunk, mark multilabel vector
#             labels_this_chunk = row[self.label_column]
            
#             # Extract label
#             labels_this_chunk_list = ast.literal_eval(labels_this_chunk)

#             # Go through each label index and one-hot encode
#             for lbl_idx in labels_this_chunk_list:
#                 chunk_labels[chunk_idx, lbl_idx] = 1.0
            
#         # Convert lists to tensors
#         video_feats = torch.tensor(np.stack(video_feats), dtype=torch.float32)  # (num_chunks, C, H, W)
#         audio_feats = torch.tensor(np.stack(audio_feats), dtype=torch.float32)  # (num_chunks, audio_feat_dim)
#         chunk_labels = torch.tensor(chunk_labels, dtype=torch.float32)          # (num_chunks, num_classes)

#         return {
#             'video_id': video_id,
#             'video': video_feats,
#             'audio': audio_feats,
#             'label': chunk_labels
#         }

#     def close(self):
#         self.audio_h5.close()
#         self.video_h5.close()
    

# class PerChunkAlignmentDataset(Dataset):
#     def __init__(self, 
#                 annotations_file='data/annotations.csv',
#                 audio_h5_path='data/audio_features.h5',
#                 video_h5_path='data/video_features.h5',
#                 split='train',
#                 num_classes=29):
#         """
#         Dataset for loading multimodal chunk-level features and labels.

#         Args:
#             annotations_file (str): Path to the chunk-level annotation CSV.
#             audio_h5_path (str): HDF5 file with audio features.
#             video_h5_path (str): HDF5 file with video features.
#             split (str): 'train', 'val', or 'test'
#         """
#         super(PerChunkAlignmentDataset, self).__init__()
        
#         # Save number of classes
#         self.num_classes = num_classes

#         # Load and filter by split
#         self.annotations = pd.read_csv(annotations_file)
#         self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

#         # Unique video IDs in this split
#         self.video_ids = self.annotations['video_id'].unique()

#         # Load label mapping from precomputed label_idx column
#         self.label_column = 'labels_idx'

#         # Open HDF5 files
#         self.audio_h5 = h5py.File(audio_h5_path, 'r')
#         self.video_h5 = h5py.File(video_h5_path, 'r')

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         row = self.annotations.iloc[idx]
#         video_id = row['video_id']
#         chunk_idx = int(row['chunk_idx'])

#         # Load audio and video features
#         audio_feat = torch.tensor(self.audio_h5[video_id][chunk_idx], dtype=torch.float32)  # shape: (1, 96, 64)
#         video_feat = torch.tensor(self.video_h5[video_id][chunk_idx], dtype=torch.float32)  # shape: (1, 16, 1024)
        
#         # Extract label
#         label_list = ast.literal_eval(row[self.label_column])

#         # One-hot encode the label
#         label = torch.tensor(one_hot_encode(label_list, self.num_classes), dtype=torch.float32)  # shape: (num_classes,)

#         return {
#             'video_id': video_id,
#             'chunk_idx': chunk_idx,
#             'audio': audio_feat,
#             'video': video_feat,
#             'label': label
#         }

#     def close(self):
#         """Closes HDF5 files (optional call)."""
#         self.audio_h5.close()
#         self.video_h5.close()
        