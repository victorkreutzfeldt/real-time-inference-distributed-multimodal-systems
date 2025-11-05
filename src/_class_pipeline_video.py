import numpy as np

import torch
import torch.nn as nn
from torch import hub

import torchvision.models as models
from torchvision import transforms
from torchvision.models import VGG19_Weights

# ======= Video =======
class VGGVideo(nn.Module):
    """
    Load VGG19 model and extract features from 'block5_pool'.
    """
    def __init__(self):
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

    def forward(self, x, return_feats=True, return_embs=True):
        # Expected x shape: (B, C, H, W) as (B, 3, 224, 224)

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
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(PostprocessorVideo, self).__init__()
        pass # TODO

    #     # Create empty matrix, for user's state_dict to load
    #     self.pca_eigen_vectors = torch.empty(
    #         (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
    #         dtype=torch.float,
    #     )

    #     self.pca_means = torch.empty(
    #         (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
    #     )

    #     self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
    #     self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    # def postprocess(self, embeddings_batch):
    #     """Applies tensor postprocessing to a batch of embeddings.

    #     Args:
    #       embeddings_batch: An tensor of shape [batch_size, embedding_size]
    #         containing output from the embedding layer of VGGish.

    #     Returns:
    #       A tensor of the same shape as the input, containing the PCA-transformed,
    #       quantized, and clipped version of the input.
    #     """
    #     assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
    #         embeddings_batch.shape,
    #     )
    #     assert (
    #         embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
    #     ), "Bad batch shape: %r" % (embeddings_batch.shape,)

    #     # Apply PCA
    #     # - Embeddings come in as [batch_size, embedding_size]
    #     # - Transpose to [embedding_size, batch_size]
    #     # - Subtract pca_means column vector from each column
    #     # - Premultiply by PCA matrix of shape [output_dims, input_dims]
    #     #   where both are are equal to embedding_size in our case
    #     # - Transpose result back to [batch_size, embedding_size]
    #     pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

    #     # Quantize by:
    #     # - clipping to [min, max] range
    #     clipped_embeddings = torch.clamp(
    #         pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
    #     )

    #     # - convert to 8-bit in range [0.0, 255.0]
    #     quantized_embeddings = torch.round(
    #         (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
    #         * (
    #             255.0
    #             / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
    #         )
    #     )

    #     return torch.squeeze(quantized_embeddings)

    # def forward(self, x):
    #     return self.postprocess(x)


class VideoPipeline(VGGVideo):
    """

    """
    def __init__(self, pretrained=True, device=None, preprocess=False, postprocess=False, progress=True):
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


    def forward(self, x, return_feats=False, return_embs=False):
        if self.preprocess:
            x = self._preprocess(x)
        x = x.to(self.device)
        out = VGGVideo.forward(self, x, return_feats, return_embs)
        if self.postprocess and self.return_embs:
            out['embs_pca'] = self._postprocess(out['embs'])
        return out

    def _preprocess(self, x):
        x = torch.stack([self._transform(xi) for xi in x]) 

        return x

    def _postprocess(self, x):
        return self.pproc(x)