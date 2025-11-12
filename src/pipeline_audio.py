# src/pipeline_audio.py

"""
Audio feature extraction pipeline based on VGGish architecture.

This module provides classes and functions to process raw audio into log-mel
spectrogram inputs, extract deep learned embeddings using a VGG-like CNN (VGGAudio),
and perform PCA whitening and quantization postprocessing consistent with AudioSet.

Key components:
    - VGGAudio: CNN feature extractor producing intermediate features and embeddings.
    - PostprocessorAudio: Applies PCA and quantization transforms preserving gradients.
    - AudioPipeline: High-level pipeline wrapping VGGAudio with optional preprocessing
    (waveform to examples) and postprocessing steps, supporting pretrained weights.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11

Notes:
    Code adapted from: https://github.com/harritaylor/torchvggish/
"""

import numpy as np

import torch
import torch.nn as nn
from torch import hub

from . import vggish_params
from . import vggish_input


class VGGAudio(nn.Module):
    """
    VGG-like convolutional neural network for audio feature extraction.

    Processes log-mel spectrogram inputs as images and outputs intermediate features and/or embeddings.

    Args:
        features (nn.Module): Feature extraction backbone (e.g., CNN layers).
    """
    def __init__(self, features: nn.Module):
        super(VGGAudio, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor, return_feats: bool = True, return_embs: bool = True) -> dict:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 96, 64) representing log-mel spectrograms.
            return_feats (bool): Whether to return intermediate features.
            return_embs (bool): Whether to return embeddings.

        Returns:
            dict: Dictionary with keys possibly including:
                'feats': intermediate features tensor.
                'embs': embeddings tensor.
        
        Raises:
            RuntimeError: If neither features nor embeddings are requested.
        """
        # Extract features treating log-mel as an image of (1, 96, 64)
        x = self.features(x) # (512, 6, 4)
        
        out = {}
        if return_feats:
            out['feats'] = x
            
        if return_embs:
            # Transpose the output from features to remain compatible with vggish embeddings
            x = torch.transpose(x, 1, 3) # (4, 6, 512)
            x = torch.transpose(x, 1, 2) # (6, 4, 512)
        
            x = x.contiguous()
            x = x.view(x.size(0), -1) # (10, 12288)

            # Extract embeddings       
            x = self.embeddings(x) # (10, 128)

            out['embs'] = x
            
        if not out:
            raise RuntimeError("AudioPipeline returns nothing!")

        return out


class PostprocessorAudio(nn.Module):
    """
    Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """
    def __init__(self):
        """
        Initializes PCA matrices and parameters as non-trainable parameters.
        """
        super(PostprocessorAudio, self).__init__()
        
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )

        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies PCA whitening and quantization to a batch of embeddings.

        Args:
            embeddings_batch (torch.Tensor): Tensor of shape (batch_size, embedding_size).

        Returns:
            torch.Tensor: Processed embeddings tensor with the same shape.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA
        # - Embeddings come in as [batch_size, embedding_size]
        # - Transpose to [embedding_size, batch_size]
        # - Subtract pca_means column vector from each column
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case
        # - Transpose result back to [batch_size, embedding_size]
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )

        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )

        return torch.squeeze(quantized_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwards input through postprocessing (calls `postprocess`).

        Args:
            x (torch.Tensor): Input embeddings tensor.

        Returns:
            torch.Tensor: Postprocessed embeddings.
        """
        return self.postprocess(x)


def make_layers() -> nn.Sequential:
    """
    Constructs the convolutional layers for the VGG audio feature extractor.

    Returns:
        nn.Sequential: Layer sequence with convs, ReLUs, and max-pooling as defined by VGGish architecture.
    """
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AudioPipeline(nn.Module):
    """
    Audio feature extraction pipeline wrapping VGGAudio, with optional preprocessing and postprocessing.

    Downloads pretrained weights if requested.

    Args:
        pretrained (bool): Whether to load pretrained VGGish weights.
        device (torch.device or None): Device to load model onto; auto-selects if None.
        preprocess (bool): Whether to preprocess raw input waveforms into examples.
        postprocess (bool): Whether to apply PCA whitening and quantization postprocessing.
        progress (bool): Show download progress when loading pretrained weights.
    """
    def __init__(self, pretrained: bool = True, device: torch.device = None, preprocess: bool = False, postprocess: bool = False, progress: bool = True):
        super().__init__()

        self.vgg = VGGAudio(make_layers())

        if pretrained:
            urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                'releases/download/v0.1/vggish_pca_params-970ea276.pth'
            }

        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            self.vgg.load_state_dict(state_dict)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.preprocess = preprocess
        self.postprocess = postprocess

        if self.postprocess:
            self.pproc = PostprocessorAudio()
            if pretrained:
                state_dict = hub.load_state_dict_from_url(urls['pca'], progress=progress)
                
                # TODO: Convert the state_dict to torch
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )

                state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)

        self.to(self.device)

    def forward(self, x, fs=None, return_feats=False, return_embs=False) -> dict:
        """
        Processes input audio through the VGGish pipeline.

        Args:
            x (numpy.ndarray or str or torch.Tensor): Raw waveform array, filename, or tensor.
            fs (int or None): Sampling frequency if `x` is a waveform array.
            return_feats (bool): Whether to return intermediate features.
            return_embs (bool): Whether to return embeddings.

        Returns:
            dict: Keys include 'feats', 'embs', optionally 'embs_pca' if postprocessing enabled.
        
        Raises:
            AttributeError: If `x` is not supported type for preprocessing.
        """
        if self.preprocess:
            x = self._preprocess(x, fs)

        x = x.to(self.device)

        out = self.vgg(x, return_feats, return_embs)

        if self.postprocess and return_embs:
            out['embs_pca'] = self._postprocess(out['embs'])
            
        return out

    def _preprocess(self, x, fs):
        """
        Preprocess raw audio input to log-mel spectrogram examples.

        Args:
            x (numpy.ndarray or str): Raw waveform or filename.
            fs (int): Sampling frequency.

        Returns:
            torch.Tensor: Processed log-mel examples.
        """
        if isinstance(x, np.ndarray):
            x = vggish_input.waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = vggish_input.wavfile_to_examples(x)
        else:
            raise AttributeError("Unsupported type for preprocessing")
        return x

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies PCA whitening and quantization postprocessing on embeddings.

        Args:
            x (torch.Tensor): Embeddings tensor.

        Returns:
            torch.Tensor: Postprocessed embeddings tensor.
        """
        return self.pproc(x)