# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
import numpy as np
import torch
from torch import nn

import losses_dai

from imu_models import LIMUBertModel4Pretrain
from typing import NamedTuple
import json
import os

import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from st_gcn import Model
from UT_HAR_model import *
from scipy.signal import stft
import matplotlib.pyplot as plt
import psutil

import tempfile
import time

def get_model_storage_size(model):
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)
        size_in_bytes = os.path.getsize(tmp_file.name)
        size_in_mb = size_in_bytes / (1024 ** 2)  # Convert to MB
    return size_in_mb

def measure_latency_and_memory(model, inputs):
    model.eval()
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = [i.to(device) for i in inputs]

    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model(*inputs)
    end_time = time.time()
    latency = end_time - start_time

    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Memory usage in MB

    return latency, memory_usage

def naive_spectrum(csi_data, sample_rate, visible):
    """
    Calculate the STFT spectrum of CSI data.

    Parameters:
    csi_data : torch.Tensor
        CSI data used for STFT spectrum generation with shape [T, S, A, L].
    sample_rate : int
        Determines the resolution of the time-domain and frequency-domain.
    visible : bool
        If True, visualize the STFT.

    Returns:
    numpy.ndarray
        Generated STFT spectrum with shape [sample_rate/2, T].
    """
    # csi_data = csi_data.mean(dim=2).unsqueeze(3)
    # Conjugate multiplication and averaging across dimensions 1, 2, 3
    csi_data = torch.mean(csi_data * torch.conj(csi_data), dim=(1, 2, 3))
    print(csi_data.shape)

    # Convert PyTorch tensor to NumPy array for STFT
    csi_data_np = csi_data.cpu().numpy()

    # Calculate the STFT
    f, t, Zxx = stft(csi_data_np, fs=sample_rate)
    print(np.abs(Zxx).flatten().shape)

    if visible:
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Magnitude')
        plt.savefig('spectrogram.png')
        ceshi

    return torch.tensor(np.abs(Zxx))

class CustomCNNMapper(nn.Module):
    def __init__(self):
        super(CustomCNNMapper, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=(1, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 1, (1, 1))

    def forward(self, x):
        # Reshape input to [batch_size, channels, height, width]
        # x = x.to(torch.float16)
        x = x.view(x.size(0), 1, 250, 270)

        # Apply convolutional layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        # x = x.to(torch.float32)

        # No need to reshape, as the output will be [256, 1, 250, 90]
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class SharedMLP(nn.Module):
    """Shared MLP for combining features from different modalities."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.shared_mlp(x)

class PretrainModelConfig(NamedTuple):
    "Configuration for BERT model"
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    feature_num: int = 0  # Factorized embedding parameterization

    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = True

    @classmethod
    def from_json(cls, js):
        return cls(**js)

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 # imu
                 imu_width: int,
                 imu_encoder: str,
                 # skel
                 skel_encoder: str, 
                 # text
                 context_length: int,
                 mlp_width: int, 
                 mlp_hidden_dim: int, 
                 mlp_layers: int,
                 step: int,
                 freeze_pretrained_encoders=False,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.freeze_pretrained_encoders = freeze_pretrained_encoders   
        self.shared_mlp = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M

        # Initialize model with the best available weights
        weights = R3D_18_Weights.DEFAULT
        vision_model = r3d_18(weights=weights)
        num_features = vision_model.fc.in_features # 512
        vision_model.fc = torch.nn.Identity()
        vision_model.eval()
        self.visual = vision_model        
        if freeze_pretrained_encoders:
            for param in self.visual.parameters():
                param.requires_grad = False
        self.mlp_image = MLP(num_features, mlp_hidden_dim, embed_dim, mlp_layers)
        self.image_projection = torch.nn.Linear(embed_dim, embed_dim)

        # Load the pre-trained LIMU_BERT model for the IMU encoder
        model_config_all = json.load(open('../offline_expand/limu_bert.json', "r"))
        name = "base_v1"
        model_cfg = PretrainModelConfig.from_json(model_config_all[name])
        self.imu_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
        self.imu_model.eval() # evaluation mode
        self.imu_model.load_state_dict(torch.load(imu_encoder + '.pt'))        
        if freeze_pretrained_encoders:
            for param in self.imu_model.parameters():
                param.requires_grad = False
        self.mlp_imu = MLP(mlp_width, mlp_hidden_dim, embed_dim, mlp_layers)   
        self.imu_projection = torch.nn.Linear(embed_dim, embed_dim)

        # Initialize model with the best available weights
        skeleton_model = Model(in_channels=3, num_class=60, dropout=0.5, edge_importance_weighting=True)
        # skeleton_model = Model(in_channels=3, num_class=400, edge_importance_weighting=True)
        weights = torch.load(skel_encoder)
        skeleton_model.load_state_dict(weights)
        num_features = 60 #400
        skeleton_model.eval()
        self.skeleton_model = skeleton_model        
        if freeze_pretrained_encoders:
            for param in self.skeleton_model.parameters():
                param.requires_grad = False
        # self.adaptation_layer = torch.nn.Linear(num_features, 60)
        self.mlp_skeleton = MLP(60, mlp_hidden_dim, embed_dim, mlp_layers)
        self.skel_projection = torch.nn.Linear(embed_dim, embed_dim)

        wisppn = UT_HAR_ViT()
        # wisppn_state_dict = torch.load('../offline_expandmmwave/UT_HAR_ViT.pt')
        # wisppn.load_state_dict(wisppn_state_dict, strict=False)
        self.mlp_csi = MLP(900, mlp_hidden_dim, embed_dim, mlp_layers)
        # wisppn = ResNet(ResidualBlock, [2, 2, 2, 2])
        # wisppn = torch.load('../offline_sing/wisppn-20190226.pkl')
        wisppn = wisppn.cuda().eval()
        self.wisppn = wisppn
        for param in self.wisppn.parameters():
            param.requires_grad = False
        # self.mlp_csi = MLP(2*18*18, mlp_hidden_dim, embed_dim, mlp_layers)
        self.csi_projection = torch.nn.Linear(embed_dim, embed_dim)
        # Add the custom CNN mapper
        self.custom_cnn_mapper = CustomCNNMapper()

        # csi = UT_HAR_CNN_GRU()
        # # csi_state_dict = torch.load('../WiFi-CSI-Sensing-Benchmark/UT_HAR_CNN+GRU.pt')
        # # csi.load_state_dict(csi_state_dict, strict=False)
        # csi = csi.cuda().eval()
        # self.csi = csi
        # for param in self.csi.parameters():
        #     param.requires_grad = False
        # self.mlp_csi2 = MLP(128, mlp_hidden_dim, embed_dim, mlp_layers)
        # self.csi_projection2 = torch.nn.Linear(embed_dim, embed_dim)

        self.distill_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.step = step
    
    def encode_csi(self, csis):
        
        x = torch.cat(csis)
        x = self.mlp_csi(x)
        x = self.shared_mlp(x)
        if self.step == 1 or self.step == 2:
            x = self.csi_projection(x)
        return x
    
    def encode_stft(self, csis):
        embed = []
        for x in csis:
            x = x.mean(dim=2)
            x = x.view(x.size(0),-1)
            x = self.csi(x.unsqueeze(0).unsqueeze(0))
            embed.append(x)
        x = torch.cat(embed)
        x = self.mlp_csi2(x)
        x = self.shared_mlp(x)
        if self.step == 1 or self.step == 2:
            x = self.csi_projection2(x)
        return x

    def encode_skeleton(self, skeletons):
        x = torch.cat(skeletons)
        x = self.mlp_skeleton(x)
        x = self.shared_mlp(x)
        if self.step == 1 or self.step == 2:
            x = self.skel_projection(x)
        return x
    
    def get_gradients(self, layer_name):
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if layer_name in name and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        return total_grad_norm ** 0.5
    
    # def forward(self, csi):
    #     csi_embed = self.encode_csi(csi)

    #     return {#'image_embed': image_embed,
    #             # 'imu_embed': imu_embed,
    #             'csi_embed': csi_embed,
    #             'logit_scale': self.logit_scale.exp()}
    
    def forward(self, csi, skeleton):
        csi_embed = self.encode_csi(csi)
        skeleton_embed = self.encode_skeleton(skeleton)

        return {'csi_embed': csi_embed,
                'skeleton_embed': skeleton_embed,
                'logit_scale': self.logit_scale.exp()}

def get_loss(model, ssl_temp, ssl_scale):
    if model.startswith('SLIP'):
        ssl_loss = losses_dai.SIMCLRLoss(temperature=ssl_temp)
        return losses_dai.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return losses_dai.CLIPLoss()
    if model.startswith('SIMCLR'):
        return losses_dai.SIMCLRLoss(temperature=ssl_temp)


def get_metric_names(model):
    if model.startswith('SLIP'):
        return ['loss', 'clip_loss', 'ssl_loss', 'clip_acc', 'ssl_acc']
    elif model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    else:
        return ['loss', 'ssl_loss', 'ssl_acc']
    
def CLIP_VITS16(ssl_mlp_dim=None, ssl_emb_dim=None, step=None, **kwargs):

    imu_encoder = '../offline_expand/hhar'
    skel_encoder = '../offline_expand/st_gcn.ntu-xsub.pt'
    model = CLIP(embed_dim=512, vision_width=400, imu_width=72, imu_encoder=imu_encoder, skel_encoder=skel_encoder, context_length=77, 
        mlp_width=72, mlp_hidden_dim=256, mlp_layers=2, step=step,
             freeze_pretrained_encoders=True, **kwargs)

    return model
    

