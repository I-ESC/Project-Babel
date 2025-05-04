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

import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from st_gcn import Model

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
        self.shared_mlp_vision = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M
        self.shared_mlp_imu = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M
        self.shared_mlp_skeleton = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M

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
        model_config_all = json.load(open('limu_bert.json', "r"))
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
        weights = torch.load(skel_encoder)
        skeleton_model.load_state_dict(weights)
        num_features = 60
        skeleton_model.eval()
        self.skeleton_model = skeleton_model        
        if freeze_pretrained_encoders:
            for param in self.skeleton_model.parameters():
                param.requires_grad = False
        self.mlp_skeleton = MLP(num_features, mlp_hidden_dim, embed_dim, mlp_layers)
        self.skel_projection = torch.nn.Linear(embed_dim, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.step = step
        # if self.step == 2:
        #     # for param in self.mlp_image.parameters():
        #     #     param.requires_grad = False
        #     for param in self.mlp_imu.parameters():
        #         param.requires_grad = False     
        #     for param in self.mlp_skeleton.parameters():
        #         param.requires_grad = False 


    def encode_image(self, images):
        # x = self.mlp_image(images)
        # x = self.shared_mlp(x).squeeze(1)  # Make sure shared_mlp is designed for batch processing
        # x = self.image_projection(x)  # Ensure image_projection is also batch-compatible
        embed = []
        for image in images:
            x = self.mlp_image(image)
            x = self.shared_mlp(x)  # Added shared MLP
            if self.step == 1 or self.step == 2:
                x = self.image_projection(x)
            embed.append(x)
        x = torch.cat(embed)

        return x

    def encode_imu(self, imus):
        embed = []
        for imu_data in imus:
            output = self.mlp_imu(imu_data).squeeze(0) # torch.Size([162, 512])
            output = self.shared_mlp(output)  # Added shared MLP
            if self.step == 1 or self.step == 2:
                output = self.imu_projection(output)

            # Compute the mean along dimension 1
            averaged_output = output.mean(dim=0) # torch.Size([512])

            embed.append(averaged_output)
        x = torch.stack(embed) # torch.Size([64, 512])

        return x

    def encode_skeleton(self, skeletons):
        # x = self.mlp_skeleton(skeletons)
        # x = self.shared_mlp(x).squeeze(1)  # Added shared MLP
        # x = self.skel_projection(x)
        embed = []
        for skeleton in skeletons:
            x = self.mlp_skeleton(skeleton)
            x = self.shared_mlp(x)  # Added shared MLP
            if self.step == 1 or self.step == 2:
                x = self.skel_projection(x)
            embed.append(x)
        x = torch.cat(embed)

        return x

    def get_gradients(self, layer_name):
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if layer_name in name and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        return total_grad_norm ** 0.5
    
    def forward(self, image, imu_data, skeleton):
        if self.step == 1: 
            imu_embed = self.encode_imu(imu_data)
            skeleton_embed = self.encode_skeleton(skeleton)
            return {'imu_embed': imu_embed,
                    'skeleton_embed': skeleton_embed,
                    'logit_scale': self.logit_scale.exp()}
        if self.step == 2:
            image_embed = self.encode_image(image)
            skeleton_embed = self.encode_skeleton(skeleton)
            return {'image_embed': image_embed,
                    'skeleton_embed': skeleton_embed,
                    'logit_scale': self.logit_scale.exp()}
        if self.step == 3:
            image_embed = self.encode_image(image)
            imu_embed = self.encode_imu(imu_data)
            skeleton_embed = self.encode_skeleton(skeleton)
            return {'image_embed': image_embed,
                    'imu_embed': imu_embed,
                    'skeleton_embed': skeleton_embed,
                    'logit_scale': self.logit_scale.exp()}

def get_loss(model, ssl_temp, ssl_scale, step):
    if model.startswith('SLIP'):
        ssl_loss = losses_dai.SIMCLRLoss(temperature=ssl_temp)
        return losses_dai.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return losses_dai.CLIPLoss(step)
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
    imu_encoder = 'hhar'
    skel_encoder = 'st_gcn.ntu-xsub.pt'
    model = CLIP(embed_dim=512, vision_width=400, imu_width=72, imu_encoder=imu_encoder, skel_encoder=skel_encoder, context_length=77, 
        mlp_width=72, mlp_hidden_dim=256, mlp_layers=2, step=step,
             freeze_pretrained_encoders=True, **kwargs)

    return model
    


