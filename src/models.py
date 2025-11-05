import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str):
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    return nn.ReLU()


class _ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.2, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout_p)
        self.act = _act(activation)

    def forward(self, x):
        h = self.act(self.ln1(self.fc1(x)))
        h = self.drop(self.act(self.ln2(self.fc2(h))))
        return x + h

# ========================== Models ==========================
class PerVideoBiLSTMAudioClassifier(nn.Module):
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
        
    def forward(self, x):
        
        feats, _ = self.lstm(x)

        if self.style == "res":
            h = self.act0(self.ln0(self.in_proj(feats)))
            h = self.residual_block(h)
            out = self.out(h)
            return out
        
        return self.classifier(feats)


class PerVideoBiLSTMVideoClassifier(nn.Module):
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
        
    def forward(self, x):
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
        
    def forward(self, audio, video):
        
        if video.dim() == 5:
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
    Audio-Guided Visual Attention with temperature control and debugging hooks.
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

    def forward(self, audio, video):

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

    # Add method to update temperature during training
    def set_temperature(self, new_temp):
        self.temperature = new_temp


class PerVideoBiLSTMAGVisualAttnMultimodalClassifier(nn.Module):
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
    
        # AG attention: process each chunk for each video
        video_attn = []
        for t in range(video.size(1)):
            attn_t = self.agva_attn(audio[:,t], video[:,t])
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






























# class PerVideoShallowAudioClassifier(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=512, num_layers=1, num_classes=29):
#         super().__init__()

#         # Temporal modelling over chunks
#         self.gru_audio = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):

#         # Apply temporal modelling over T
#         gru_out, _ = self.gru_audio(x) 

#         # Apply classifier
#         out = self.classifier(gru_out)  # (B, T, num_classes)
        
#         return out


# class PerVideoShallowVideoClassifier(nn.Module):
#     def __init__(self, input_dim=512*7*7, hidden_dim=512, num_layers=1, num_classes=29):
#         super().__init__()

#         # Temporal modelling over chunks
#         self.gru_video = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):

#         # Extract dimensions
#         B, T = x.size(0), x.size(1)

#         # Flatten video spatial dimensions
#         x = x.view(B, T, -1)

#         # Apply temporal modelling over T
#         gru_out, _ = self.gru_video(x)  # (batch, N, hidden_dim)

#         # Apply classifier
#         out = self.classifier(gru_out)  # (batch, N, num_classes)

#         return out


# class PerVideoShallowMultimodalClassifier(nn.Module):
#     """
    
#     """
#     def __init__(self, audio_dim=128, video_dim=512*7*7, hidden_dim=512, num_classes=29):
#         super().__init__()

#         # Temporal modelling over chunks
#         self.gru_audio = nn.GRU(audio_dim, hidden_dim, batch_first=True)
#         self.gru_video = nn.GRU(video_dim, hidden_dim, batch_first=True)
        
#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, audio, video):

#         # Extract dimensions
#         B, T = video.size(0), video.size(1)

#         # Flatten video spatial dimensions
#         video = video.view(B, T, -1)

#         # Apply temporal modelling over T
#         gru_out_audio, _ = self.gru_audio(audio)
#         gru_out_video, _ = self.gru_video(video)
        
#         # Fusion with concatenation
#         out = torch.cat([gru_out_audio, gru_out_video], dim=-1)

#         # Apply classifier
#         out = self.classifier(out)

#         return out


# class PerVideoShallowMultimodalClassifierGAP(nn.Module):
#     """
#     GAP variant:
#       - Accepts video as (B, T, 512, 7, 7) or (B, T, 512).
#       - If 5D, applies GAP over spatial dims -> (B, T, 512).
#       - GRU for video expects 512-d inputs.
#       - Audio path unchanged.
#     """
#     def __init__(self, audio_dim=128, video_dim=512, hidden_dim=512, num_classes=29):
#         super().__init__()
#         assert video_dim == 512, "This GAP variant is designed for video_dim=512 after spatial pooling."
#         self.video_dim = video_dim

#         # Temporal modelling over chunks
#         self.gru_audio = nn.GRU(audio_dim, hidden_dim, batch_first=True)
#         self.gru_video = nn.GRU(video_dim, hidden_dim, batch_first=True)

#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, audio, video):
#         # video can be (B,T,512,7,7) or (B,T,512)
#         if video.dim() == 5:
#             # GAP over spatial dims H,W
#             video = video.mean(dim=(-1, -2))  # (B, T, 512)
#         elif video.dim() == 3:
#             # already (B, T, 512), pass-through
#             pass
#         else:
#             raise ValueError(f"Unexpected video shape: {tuple(video.shape)}")
#         #breakpoint()
#         # Sanity check
#         if video.size(-1) != self.video_dim:
#             raise ValueError(f"video feature dim {video.size(-1)} != expected {self.video_dim}")

#         # GRUs over time
#         gru_out_audio, _ = self.gru_audio(audio)   # (B, T, H)
#         gru_out_video, _ = self.gru_video(video)   # (B, T, H)

#         # Late fusion (concat) per time step
#         out = torch.cat([gru_out_audio, gru_out_video], dim=-1)  # (B, T, 2H)

#         # Classifier head per time step
#          wiout = self.classifier(out)  # (B, T, num_classes)
#         return out


# class PerVideoAGVisualAttMultimodalClassifier(nn.Module):
#     def __init__(self, audio_dim=128, video_dim=512 * 7 * 7, hidden_dim=512, emb_att_dim=128, num_classes=29):
#         super().__init__()

#         # Temporal modelling over chunks
#         self.gru_audio = nn.GRU(audio_dim, hidden_dim, batch_first=True)
#         self.gru_video = nn.GRU(512, hidden_dim, batch_first=True)

#         # Audio-guided visual attention
#         self.agva_attn = AGVAttn(emb_att_dim=emb_att_dim, num_visual_regions=49)

#         # Classifier head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, audio_feats, video_feats):
        
#         # Apply audio guided attention
#         video_attn = self.agva_attn(audio_feats, video_feats)

#         # Process audio and video with their respective LSTMs
#         gru_out_audio, _ = self.gru_audio(audio_feats)
#         gru_out_video, _ = self.gru_video(video_attn)

#         # Apply audio and video fusion using concatenation
#         fusion = torch.cat((gru_out_audio, gru_out_video), dim=-1)    

#         # Pass through the classifier
#         out = self.classifier(fusion)

#         return out

# class AGVAttn(nn.Module):
#     """
#     AGVAttn: Audio-Guided Visual Attention 

#     """

#     def __init__(
#             self, emb_att_dim: int = 512, num_visual_regions: int = 49, init_weights: bool = True
#             ) -> None:
#         super(AGVAttn, self).__init__()

#         self.projection_audio = nn.Sequential(
#                 nn.Linear(128, emb_att_dim),
#                 nn.ReLU()
#             )
        
#         self.projection_video = nn.Sequential(
#                 nn.Linear(512, emb_att_dim),
#                 nn.ReLU()
#             )
        
#         self.fusion_audio = nn.Linear(emb_att_dim, num_visual_regions, bias=False)
#         self.fusion_video = nn.Linear(emb_att_dim, num_visual_regions, bias=False)

#         self.tanh = nn.Tanh()

#         self.attn_scores = nn.Linear(num_visual_regions, 1, bias=False)

#         if init_weights:
#             torch.nn.init.xavier_uniform_(self.projection_audio[0].weight)
#             torch.nn.init.xavier_uniform_(self.projection_video[0].weight)

#             torch.nn.init.xavier_uniform_(self.fusion_audio.weight)
#             torch.nn.init.xavier_uniform_(self.fusion_video.weight)
#             torch.nn.init.xavier_uniform_(self.attn_scores.weight)

#     def forward(self, audio, video):

#         # Get duration
#         duration = video.size(1)

#         # Ignore temporal dimension
#         audio = audio.view(-1, 128)
#         video = video.reshape((video.size(0) * video.size(1), -1, 512))
        
#         # Project audio
#         audio_proj = self.projection_audio(audio)
#         video_proj = self.projection_video(video)

#         # Fusion audio and video
#         fusion = self.fusion_audio(audio_proj).unsqueeze(2) + self.fusion_video(video_proj)
#         fusion = self.tanh(fusion)

#         # Compute scores
#         scores = self.attn_scores(fusion)
#         scores = scores.squeeze()

#         # Compute attention
#         attn = F.softmax(scores, dim=-1)
#         attn = attn.unsqueeze(1)

#         # Compute audio-guided video
#         video_attn = torch.bmm(attn, video)
#         video_attn = video_attn.squeeze()
#         video_attn = video_attn.view(-1, duration, 512)

#         return video_attn








# class ShallowMultimodalClassifier(nn.Module):
#     def __init__(self, audio_dim=128, video_dim=512 * 7 * 7, num_classes=29):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(audio_dim + video_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, audio_feat, video_feat):
#         audio_feat = audio_feat.view(audio_feat.size(0), -1)  # (B, 128)
#         video_feat = video_feat.view(video_feat.size(0), -1)  # (B, 25088)
#         fused = torch.cat((audio_feat, video_feat), dim=1)    # (B, 25216)
#         return self.classifier(fused)


# class DummyMultimodalClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(128 + 512, 10)

#     def forward(self, audio_feat, video_feat):
#         x = torch.cat([audio_feat.mean(dim=0), video_feat.mean(dim=0)], dim=-1)
#         return self.fc(x.unsqueeze(0))


# class EarlyFusionGRU(nn.Module):
#     """
    
#     """
#     def __init__(self, audio_dim=128, video_dim=512*7*7, hidden_dim=256, num_classes=29):
#         super().__init__()

#         self.gru = nn.GRU(audio_dim + video_dim, hidden_dim, batch_first=True)
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, audio, video):
#         B, T = video.size(0), video.size(1)
#         video = video.view(B, T, -1)
#         x = torch.cat([audio, video], dim=-1)
#         h, _ = self.gru(x)
        
#         return self.classifier(h)











# class AGVAttnNet(nn.Module):
#     """
#     AGVAttnNet: Audio-Guided Visual Attention Network


#     """
#     def __init__(
#             self, emb_dim: int, emb_att_dim: int, hidden_dim: int, num_classes: int, device: torch.device = 'cpu',
#             init_weights: bool = True
#             ) -> None:
#         super(AGVAttnNet, self).__init__()

#         self.device = device
#         self.hidden_dim = hidden_dim

#         # Audion-Guided Visual Attention
#         self.agva_attn = AGVAttn(emb_att_dim=emb_att_dim, num_visual_regions=49)

#         # LSTM layers for audio and video modalities
#         self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
#         self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)

#         # Non-linear activation and affine transformations
#         self.relu = nn.ReLU()

#         # self.affine_v = nn.Linear(emb_att_dim, 49, bias=False)
#         # self.affine_g = nn.Linear(emb_att_dim, 49, bias=False)
#         # self.affine_h = nn.Linear(49, 1, bias=False)

#         # Fully connected layers for classification
#         self.L1 = nn.Linear(hidden_dim * 4, 64)
#         self.L2 = nn.Linear(64, num_classes)

#         # Initialize weights
#         if init_weights:

#             # init.xavier_uniform_(self.affine_v.weight)
#             # init.xavier_uniform_(self.affine_g.weight)
#             # init.xavier_uniform_(self.affine_h.weight)

#             torch.nn.init.xavier_uniform_(self.L1.weight)
#             torch.nn.init.xavier_uniform_(self.L2.weight)


#     def forward(self, audio, video):
        
#         # Apply audio guided attention
#         video_attn = self.agva_attn(audio, video)

#         # Bi-LSTM for temporal modeling
#         hidden1 = (torch.zeros(2, audio.size(0), self.hidden_dim).to(self.device),
#                    torch.zeros(2, audio.size(0), self.hidden_dim).to(self.device))
        
#         hidden2 = (torch.zeros(2, video.size(0), self.hidden_dim).to(self.device),
#                    torch.zeros(2, video.size(0), self.hidden_dim).to(self.device))
        
#         self.lstm_audio.flatten_parameters()    
#         self.lstm_video.flatten_parameters()

#         # Process audio and video with their respective LSTMs
#         lstm_audio, hidden1 = self.lstm_audio(audio.view(len(audio), 10, -1), hidden1)
#         lstm_video, hidden2 = self.lstm_video(video_attn.view(len(video), 10, -1), hidden2)
        
#         # Concatenate audio and video features and pass through fully connected layers
#         output = torch.cat((lstm_audio, lstm_video), -1)
#         output = self.relu(output)
#         out = self.L1(output)
#         out = self.relu(out)
#         out = self.L2(out)
        
#         #out = F.softmax(out, dim=-1)

#         return out
    
# class att_Net(nn.Module):
#     '''
#     Audio-visual event localization with audio-guided visual attention and audio-visual fusion.
#     '''
#     def __init__(self, emb_dim, hidden_dim, hidden_size, tagset_size, device):
#         super(att_Net, self).__init__()

#         # Keep track
#         self.hidden_dim = hidden_dim
#         self.device = device
        
#         # LSTM layers for audio and video modalities
#         self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
#         self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)

#         # Non-linear activation and affine transformations
#         self.relu = nn.ReLU()
#         self.affine_audio = nn.Linear(128, hidden_size)
#         self.affine_video = nn.Linear(512, hidden_size)
#         self.affine_v = nn.Linear(hidden_size, 49, bias=False)
#         self.affine_g = nn.Linear(hidden_size, 49, bias=False)
#         self.affine_h = nn.Linear(49, 1, bias=False)

#         # Fully connected layers for classification
#         self.L1 = nn.Linear(hidden_dim * 4, 64)
#         self.L2 = nn.Linear(64, tagset_size)

#         # Initialize weights
#         self.init_weights()

#     def init_weights(self):
#         """Initialize the weights."""
#         torch.nn.init.xavier_uniform_(self.affine_v.weight)
#         torch.nn.init.xavier_uniform_(self.affine_g.weight)
#         torch.nn.init.xavier_uniform_(self.affine_h.weight)
#         torch.nn.init.xavier_uniform_(self.L1.weight)
#         torch.nn.init.xavier_uniform_(self.L2.weight)
#         torch.nn.init.xavier_uniform_(self.affine_audio.weight)
#         torch.nn.init.xavier_uniform_(self.affine_video.weight)

#     def forward(self, audio, video):
#         # Prepare the video data
#         v_t = video.view(video.size(0) * video.size(1), -1, 512)
#         V = v_t

#         # Audio-guided visual attention
#         v_t = self.relu(self.affine_video(v_t))
#         a_t = audio.view(-1, audio.size(-1))
#         a_t = self.relu(self.affine_audio(a_t))

#         # Fusion of audio and visual features
#         content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2)
#         z_t = self.affine_h(F.tanh(content_v)).squeeze(2)

#         # Attention
#         alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))
#         c_t = torch.bmm(alpha_t, V).view(-1, 512)
#         video_t = c_t.view(video.size(0), -1, 512)

#         # Bi-LSTM for temporal modeling
#         hidden1 = (torch.zeros(2, audio.size(0), self.hidden_dim).to(self.device),
#                     torch.zeros(2, audio.size(0), self.hidden_dim).to(self.device))
        
#         hidden2 = (torch.zeros(2, video.size(0), self.hidden_dim).to(self.device),
#                     torch.zeros(2, video.size(0), self.hidden_dim).to(self.device))
        
#         self.lstm_audio.flatten_parameters()    
#         self.lstm_video.flatten_parameters()

#         # Process audio and video with their respective LSTMs
#         lstm_audio, hidden1 = self.lstm_audio(audio.view(len(audio), 10, -1), hidden1)
#         lstm_video, hidden2 = self.lstm_video(video_t.view(len(video), 10, -1), hidden2)
        
#         # Concatenate audio and video features and pass through fully connected layers
#         output = torch.cat((lstm_audio, lstm_video), -1)
#         output = self.relu(output)
#         out = self.L1(output)
#         out = self.relu(out)
#         out = self.L2(out)
        
#         #out = F.softmax(out, dim=-1)
#         #breakpoint()

#         return out