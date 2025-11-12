# src/pipeline_video.py

"""
Video feature extraction pipeline based on pretrained VGG19 CNN.

This module provides classes to extract convolutional features and dense embeddings
from video frames using a VGG19 backbone truncated at the fifth max pooling layer.

Key components:
    - VGGVideo: CNN feature extractor producing intermediate features and dense embeddings.
    - PostprocessorVideo: Placeholder for post-processing such as PCA whitening and quantization (currently TODO).
    - VideoPipeline: Integration of VGGVideo with optional preprocessing (normalization)
    and postprocessing steps, supporting device selection and pretrained weights.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import numpy as np

import torch
import torch.nn as nn
from torch import hub

import torchvision.models as models
from torchvision import transforms
from torchvision.models import VGG19_Weights


class VGGVideo(nn.Module):
    """
    Pretrained VGG19-based CNN feature extractor for video frames.

    Extracts intermediate convolutional features and dense embeddings from video inputs.

    The embeddings are produced by flattening the convolutional output and passing through
    fully connected layers with ReLU activation.
    """
    def __init__(self):
        """
        Initializes the model by loading pretrained VGG19 layers and defining embedding layers.
        """
        super(VGGVideo, self).__init__()

        # Load pre-trained VGG19 model
        weights = VGG19_Weights.IMAGENET1K_V1
        vgg19 = models.vgg19(weights=weights).features

        self.features = nn.Sequential(*list(vgg19.children())[:37])  # Up to MaxPool of the 5th max-pooling layer
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor, return_feats: bool = True, return_embs: bool = True) -> dict:
        """
        Forward pass through CNN and embedding layers.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, 3, 224, 224).
            return_feats (bool): Whether to return intermediate CNN features.
            return_embs (bool): Whether to return embeddings.

        Returns:
            dict: Dictionary containing 'feats' and/or 'embs', or raises RuntimeError if empty.
        """
        # Extract features
        x = self.features(x) # (512, 7, 7) = (C, H, W)
        
        out = {}
        if return_feats:
            out['feats'] = x

        if return_embs:
            # Transpose the output from features to remain compatible with vggish embeddings
            x = torch.transpose(x, 1, 3) # (7, 7, 512) = (W, H, C)
            x = torch.transpose(x, 1, 2) # (7, 7, 512) = (H, W, C)
        
            x = x.contiguous()
            x = x.view(x.size(0), -1) # (10, 25088)

            # Extract embeddings       
            x = self.embeddings(x) # (10, 1024)

            out['embs'] = x
            
        if not out:
            raise RuntimeError("VideoPipeline returns nothing!")
        
        return out


class PostprocessorVideo(nn.Module):
    """
    Placeholder for video embedding postprocessing (e.g., PCA whitening and quantization).
    """

    def __init__(self):
        """
        Initializes the postprocessor. TODO: Implement PCA and quantization functionality.
        """
        super(PostprocessorVideo, self).__init__()
        pass # TODO


class VideoPipeline(VGGVideo):
    """
    Video processing pipeline combining VGGVideo feature extractor with optional preprocessing
    and postprocessing steps.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        device (torch.device or None): Device to move model and data to. Auto-detects if None.
        preprocess (bool): Whether to apply input preprocessing (normalization, tensor conversion).
        postprocess (bool): Whether to apply postprocessing on embeddings (currently TODO).
        progress (bool): Show progress bar when downloading pretrained weights.
    """
    def __init__(self, pretrained: bool = True, device: torch.device = None, preprocess: bool = False, postprocess: bool = False, progress: bool = True):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.preprocess = preprocess
        if self.preprocess:
            self._transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=[0.422, 0.392, 0.367], # Stats from the train dataset (data/pixel_stats.py)
                    std=[0.277, 0.275, 0.280])
            ])

        self.postprocess = postprocess

        if self.postprocess:
            self.pproc = PostprocessorVideo()
            if pretrained:
                pass
            # TODO: To-be implemented if needed.

        self.to(self.device)

    def forward(self, x, return_feats: bool = False, return_embs: bool = False) -> dict:
        """
        Processes input video batch through preprocessing, feature extraction, and optional postprocessing.

        Args:
            x (List or torch.Tensor): Batch of images as tensors or PIL images.
            return_feats (bool): Whether to return convolutional features.
            return_embs (bool): Whether to return dense embeddings.

        Returns:
            dict: Dictionary with keys 'feats', 'embs', and optionally 'embs_pca'.
        """
        if self.preprocess:
            x = self._preprocess(x)

        x = x.to(self.device)

        out = VGGVideo.forward(self, x, return_feats, return_embs)

        if self.postprocess and self.return_embs:
            out['embs_pca'] = self._postprocess(out['embs'])

        return out

    def _preprocess(self, x):
        """
        Applies preprocessing transforms to input batch: normalization and tensor conversion.

        Args:
            x (List): List of PIL.Image or numpy arrays representing images.

        Returns:
            torch.Tensor: Batch tensor of shape (batch_size, 3, H, W).
        """
        x = torch.stack([self._transform(xi) for xi in x]) 

        return x

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies postprocessing to embeddings, e.g., PCA whitening and quantization.

        Args:
            x (torch.Tensor): Embeddings tensor.

        Returns:
            torch.Tensor: Postprocessed embeddings tensor.
        """
        return self.pproc(x)