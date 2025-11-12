# src/models.py

"""
Neural network models for multimodal audio-visual event localization (AVEL).

This module defines configurable BiLSTM-based classifiers for audio, video, and
multimodal fusion tasks, featuring several classifier head styles including base,
regularized, wide, deep, and residual block variants.

Key components:
    - Activation selector utility (_act) supporting GELU, SiLU, and ReLU.
    - Residual fully connected block (_ResidualBlock) with layer norm and dropout.
    - PerVideoBiLSTMAudioClassifier: Audio-only BiLSTM classifier.
    - PerVideoBiLSTMVideoClassifier: Video-only BiLSTM classifier, supports 4D/5D inputs.
    - PerVideoBiLSTMMultimodalClassifier: Fuses audio and video streams with configurable classifier heads.
    - AGVAttn: Audio-guided visual attention module with temperature scaling and debug outputs.
    - PerVideoBiLSTMAGVisualAttnMultimodalClassifier: Multimodal model integrating audio-guided visual attention with BiLSTM fusion.

Design permits flexible architecture and activation choices for experimentation
with multimodal sequential video classification in a real-time distributed inference framework.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union


def _act(name: str) -> nn.Module:
    """
    Utility function to select activation function by name.

    Args:
        name (str): Name of the activation function. Supported: 'gelu', 'silu', others default to ReLU.

    Returns:
        nn.Module: Corresponding PyTorch activation function module.
    """
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    return nn.ReLU()


class _ResidualBlock(nn.Module):
    """
    A residual fully connected block with layer normalization, dropout, and configurable activation.

    Args:
        dim (int): Dimension of input and output embeddings.
        dropout_p (float): Dropout probability.
        activation (str): Activation function name ('gelu' or others default to ReLU).
    """
    def __init__(self, dim: int, dropout_p=0.2, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout_p)
        self.act = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        h = self.act(self.ln1(self.fc1(x)))
        h = self.drop(self.act(self.ln2(self.fc2(h))))
        return x + h


class PerVideoBiLSTMAudioClassifier(nn.Module):
    """
    BiLSTM-based audio classifier with configurable classification head styles.

    Args:
        input_dim (int): Dimensionality of input audio embeddings.
        num_classes (int): Number of output classes.
        style (str): Classification head style: 'base', 'reg', 'wide', 'deep', or 'res'.
        hidden_dim (int): Hidden LSTM dimension size.
        num_layers (int): Number of LSTM layers.
        dropout_p (float): Dropout probability for classification head.
        activation (str): Activation function name ('relu' or 'gelu').
    """
    def __init__(self, input_dim=128, num_classes=29, style="base", hidden_dim=512, num_layers=1, dropout_p=0.3, activation="relu"):
        super().__init__()
        style = style.lower()
        self.style = style
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

        # Styles
        if style == "base":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 64),
                self.activation,
                nn.Linear(64, num_classes)
            )
        elif style == "reg":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "wide":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 128),
                nn.LayerNorm(128),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(128, num_classes)
            )
        elif style == "deep":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 128),
                nn.LayerNorm(128),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "res":
            self.in_proj = nn.Linear(hidden_dim*2, 128)
            self.residual_block = _ResidualBlock(128, dropout_p=dropout_p, activation="gelu")
            self.out = nn.Linear(128, num_classes)
            self.ln0 = nn.LayerNorm(128)
            self.act0 = nn.GELU()
        else:
            raise ValueError(f"Unknown style {style}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, num_classes).
        """
        feats, _ = self.lstm(x)

        if self.style == "res":
            h = self.act0(self.ln0(self.in_proj(feats)))
            h = self.residual_block(h)
            out = self.out(h)
            return out
        
        return self.classifier(feats)


class PerVideoBiLSTMVideoClassifier(nn.Module):
    """
    BiLSTM-based video classifier with configurable classification head styles.

    Args:
        input_dim (int): Dimensionality of input video embeddings.
        num_classes (int): Number of output classes.
        style (str): Style of classification head.
        hidden_dim (int): Hidden dimension size for LSTM.
        num_layers (int): Number of LSTM layers.
        dropout_p (float): Dropout probability.
        activation (str): Activation function name.
    """
    def __init__(self, input_dim=512, num_classes=29, style="base", hidden_dim=512, num_layers=1, dropout_p=0.3, activation="relu"):
        super().__init__()
        style = style.lower()
        self.style = style
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        if style == "base":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 64),
                self.activation,
                nn.Linear(64, num_classes)
            )
        elif style == "reg":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "wide":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 256),
                nn.LayerNorm(256),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(256, num_classes)
            )
        elif style == "deep":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim*2, 256),
                nn.LayerNorm(256),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(128, num_classes)
            )
        elif style == "res":
            self.in_proj = nn.Linear(hidden_dim*2, 256)
            self.residual_block = _ResidualBlock(256, dropout_p=dropout_p, activation="gelu")
            self.out = nn.Linear(256, num_classes)
            self.ln0 = nn.LayerNorm(256)
            self.act0 = nn.GELU()
        else:
            raise ValueError(f"Unknown style {style}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor; accepts 4D or 5D video embeddings.
                              If 5D, spatial dimensions are averaged.

        Returns:
            torch.Tensor: Classification logits per token.
        """
        if x.dim() == 5:
            x = x.mean(dim=(-1, -2))
        feats, _ = self.lstm(x)

        if self.style == "res":
            h = self.act0(self.ln0(self.in_proj(feats)))
            h = self.residual_block(h)
            out = self.out(h)
            return out
        
        return self.classifier(feats)


class PerVideoBiLSTMMultimodalClassifier(nn.Module):
    """
    Multimodal BiLSTM classifier fusing audio and video streams.

    Supports various styles of classifier heads including residual blocks.

    Args:
        audio_dim (int): Audio feature dimension.
        video_dim (int): Video feature dimension.
        num_classes (int): Number of output classes.
        style (str): Style of classification head.
        hidden_dim (int): Hidden size used in LSTMs.
        num_layers (int): Number of LSTM layers.
        dropout_p (float): Dropout probability.
        activation (str): Activation function name.
    """
    def __init__(self, audio_dim=128, video_dim=512, num_classes=29, style="base", hidden_dim=512, num_layers=1, dropout_p=0.3, activation="relu"):
        super().__init__()
        style = style.lower()
        self.style = style
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.audio_lstm = nn.LSTM(audio_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.video_lstm = nn.LSTM(video_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        fusion_dim = hidden_dim * 4

        if style == "base":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                self.activation,
                nn.Linear(512, 64),
                self.activation,
                nn.Linear(64, num_classes)
            )
        elif style == "reg":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.LayerNorm(512),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(512, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "wide":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 2048),
                nn.LayerNorm(2048),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(1024, num_classes)
            )
        elif style == "deep":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 2048),
                nn.LayerNorm(2048),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(512, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "res":
            hidden1 = 2048
            hidden2 = 1024
            self.in_proj = nn.Linear(fusion_dim, hidden1)
            self.block1 = _ResidualBlock(hidden1, dropout_p=dropout_p, activation=activation)
            self.block2 = _ResidualBlock(hidden2, dropout_p=dropout_p, activation=activation)
            self.proj = nn.Linear(hidden1, hidden2)
            self.out = nn.Linear(hidden2, num_classes)
            self.ln0 = nn.LayerNorm(hidden1)
            self.act0 = _act(activation)
            self.ln1 = nn.LayerNorm(hidden2)
            self.act1 = _act(activation)
            self.classifier = None
        else:
            raise ValueError(f"Unknown style '{style}'")
        
    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): Audio input tensor (batch_size, seq_len, audio_dim).
            video (torch.Tensor): Video input tensor (batch_size, seq_len, video_dim) or 5D tensor to be spatially averaged.

        Returns:
            torch.Tensor: Logits per token of shape (batch_size, seq_len, num_classes).
        """
        if video.dim() == 5:
            video = video.mean(dim=(-1, -2))

        if video.dim() == 4:
            video = video.mean(dim=(-1, -2)) 
        
        audio_out, _ = self.audio_lstm(audio)
        video_out, _ = self.video_lstm(video)
        fusion = torch.cat([audio_out, video_out], dim=-1)
        
        if self.style == "res":
            h = self.act0(self.ln0(self.in_proj(fusion)))
            h = self.block1(h)
            h = self.act1(self.ln1(self.proj(h)))
            h = self.block2(h)
            return self.out(h)
        
        return self.classifier(fusion)


class AGVAttn(nn.Module):
    """
    Audio-Guided Visual Attention module with temperature scaling and optional debug outputs.

    Args:
        emb_att_dim (int): Dimension for projected audio/video embeddings.
        num_visual_regions (int): Number of visual spatial regions.
        temperature (float): Temperature scaling for attention logits.
        init_weights (bool): Whether to initialize weights with Xavier uniform.
        debug (bool): Enables debug outputs of attention maps.
        dropout_p (float): Dropout rate applied to projections.
    """
    def __init__(self, emb_att_dim=512, num_visual_regions=49, temperature=1.0, init_weights=False, debug=False, dropout_p=0.2):
        super().__init__()
        self.temperature = temperature
        self.debug = debug  # Flag to enable debug outputs/logging

        self.projection_audio = nn.Sequential(
            nn.Linear(128, emb_att_dim),
            nn.ReLU()
        )
        self.projection_video = nn.Sequential(
            nn.Linear(512, emb_att_dim),
            nn.ReLU()
        )

        # Optional LayerNorm and Dropout after projections
        self.ln_audio = nn.LayerNorm(emb_att_dim)
        self.ln_video = nn.LayerNorm(emb_att_dim)
        self.dropout = nn.Dropout(dropout_p)

        self.fusion_audio = nn.Linear(emb_att_dim, num_visual_regions, bias=False)
        self.fusion_video = nn.Linear(emb_att_dim, num_visual_regions, bias=False)
        
        self.act = nn.Tanh()
    
        self.attn_scores = nn.Linear(num_visual_regions, 1, bias=False)

        if init_weights:
            torch.nn.init.xavier_uniform_(self.projection_audio[0].weight)
            torch.nn.init.xavier_uniform_(self.projection_video[0].weight)
            
            torch.nn.init.xavier_uniform_(self.fusion_audio.weight)
            torch.nn.init.xavier_uniform_(self.fusion_video.weight)

            torch.nn.init.xavier_uniform_(self.attn_scores.weight)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass generating attended visual embeddings guided by audio.

        Args:
            audio (torch.Tensor): Audio embeddings (batch_size, emb_dim).
            video (torch.Tensor): Video embeddings (batch_size, channels, height, width).

        Returns:
            torch.Tensor or tuple: Attended video embeddings; if debug is True also returns attention maps and raw scores.
        """

        # TODO: Correct this model!!!

        # Transpose video original shape (B, C, H, W) to meet how audio embeddings was created,
        # that is, (B, H, W, C)
        video = torch.transpose(video, 1, 3)
        video = torch.transpose(video, 1, 2)
        
        # Now reshape such that (B, H*W, C)
        video = video.reshape(video.size(0), -1, 512)
        video = video.contiguous()
        
        audio_proj = self.projection_audio(audio)       # (B, emb_att_dim)
        #audio_proj = self.ln_audio(audio_proj)
        #audio_proj = self.dropout(audio_proj)

        video_proj = self.projection_video(video)       # (B, num_regions, emb_att_dim)
        #video_proj = self.ln_video(video_proj)
        #video_proj = self.dropout(video_proj)
                                  
        fusion = self.fusion_audio(audio_proj).unsqueeze(2) + self.fusion_video(video_proj)
        #fusion = self.act(fusion)
        
        scores = self.attn_scores(fusion).squeeze(-1)  # (B, num_regions)
        scores = scores / self.temperature

        # Apply dropout to attention logits (before softmax)
        #scores = self.dropout(scores)

        attn = F.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, num_regions)
        
        video_attn = torch.bmm(attn, video).squeeze(1)  # (B, 512)
        
        if self.debug:
            # Debug outputs strongly recommended to log/visualize
            # Return attention weights as well
            return video_attn, attn.squeeze(1), scores
        
        return video_attn

    def set_temperature(self, new_temp: float) -> None:
        """
        Update the temperature used to scale attention scores.

        Args:
            new_temp (float): New temperature value.
        """
        self.temperature = new_temp


class PerVideoBiLSTMAGVisualAttnMultimodalClassifier(nn.Module):
    """
    Multimodal classifier integrating audio-guided visual attention with BiLSTM fusion.

    Args:
        audio_dim (int): Audio input dimension.
        video_dim (int): Visual input dimension.
        emb_att_dim (int): Dimension for attention embeddings.
        num_classes (int): Output class count.
        style (str): Classification head style.
        debug (bool): Enables attention debug outputs.
        temperature (float): Temperature applied to attention logits.
        hidden_dim (int): Hidden size for LSTM layers.
        num_layers (int): Number of LSTM layers.
        dropout_p (float): Dropout rate.
        activation (str): Activation function name.
    """
    def __init__(self, audio_dim=128, video_dim=512, emb_att_dim=128, num_classes=29, style="base", debug=False, temperature=1.0, hidden_dim=512, num_layers=1, dropout_p=0.3, activation="relu"):
        super().__init__()
        style = style.lower()
        self.style = style
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.agva_attn = AGVAttn(emb_att_dim=emb_att_dim, num_visual_regions=49, debug=debug, temperature=temperature)
        self.audio_lstm = nn.LSTM(audio_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.video_lstm = nn.LSTM(video_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        fusion_dim = hidden_dim * 4
        if style == "base":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                self.activation,
                nn.Linear(512, 64),
                self.activation,
                nn.Linear(64, num_classes)
            )
        elif style == "reg":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.LayerNorm(512),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(512, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "wide":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 2048),
                nn.LayerNorm(2048),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(1024, num_classes)
            )
        elif style == "deep":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 2048),
                nn.LayerNorm(2048),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(512, 64),
                nn.LayerNorm(64),
                self.activation,
                nn.Dropout(dropout_p),
                nn.Linear(64, num_classes)
            )
        elif style == "res":
            hidden1 = 2048
            hidden2 = 1024
            self.in_proj = nn.Linear(fusion_dim, hidden1)
            self.block1 = _ResidualBlock(hidden1, dropout_p=dropout_p, activation=activation)
            self.block2 = _ResidualBlock(hidden2, dropout_p=dropout_p, activation=activation)
            self.proj = nn.Linear(hidden1, hidden2)
            self.out = nn.Linear(hidden2, num_classes)
            self.ln0 = nn.LayerNorm(hidden1)
            self.act0 = _act(activation)
            self.ln1 = nn.LayerNorm(hidden2)
            self.act1 = _act(activation)
            self.classifier = None
        else:
            raise ValueError(f"Unknown style '{style}'")
        
    def forward(self, audio, video):
        """
        Forward pass applying audio-guided visual attention and multimodal fusion.

        Args:
            audio (torch.Tensor): Audio embeddings tensor.
            video (torch.Tensor): Video embeddings tensor including sequence dimension.

        Returns:
            torch.Tensor: Output logits per token.
        """
        # AG attention: process each chunk for each video
        video_attn = []
        for t in range(video.size(1)):
            attn_t = self.agva_attn(audio[:, t], video[:, t])
            video_attn.append(attn_t)

        video_attn = torch.stack(video_attn, dim=1)
        audio_out, _ = self.audio_lstm(audio)
        video_out, _ = self.video_lstm(video_attn)
        fusion = torch.cat([audio_out, video_out], dim=-1)

        if self.style == "res":
            h = self.act0(self.ln0(self.in_proj(fusion)))
            h = self.block1(h)
            h = self.act1(self.ln1(self.proj(h)))
            h = self.block2(h)
            return self.out(h)
        
        return self.classifier(fusion)