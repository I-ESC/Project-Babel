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
from UT_HAR_model import *
from resnet2d import resnet18_mutual


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
        self.shared_mlp_wifi = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M
        self.shared_mlp_mmwave = SharedMLP(embed_dim, mlp_hidden_dim, embed_dim)  # Shared M

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
        wisppn_state_dict = torch.load('UT_HAR_ViT.pt')
        wisppn.load_state_dict(wisppn_state_dict, strict=False)
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

        csi = UT_HAR_CNN_GRU()
        # csi_state_dict = torch.load('./UT_HAR_CNN+GRU.pt')
        # csi.load_state_dict(csi_state_dict, strict=False)
        csi = csi.cuda().eval()
        self.csi = csi
        for param in self.csi.parameters():
            param.requires_grad = False
        self.mlp_csi2 = MLP(128, mlp_hidden_dim, embed_dim, mlp_layers)
        self.csi_projection2 = torch.nn.Linear(embed_dim, embed_dim)

        mmWave_model = resnet18_mutual()
        # mmWave_model.load_state_dict(torch.load('../XRF55-repo/result/params/model2_params.pth'))
        mmWave_model = mmWave_model.cuda().eval()
        self.mmWave_model = mmWave_model
        for param in self.mmWave_model.parameters():
            param.requires_grad = False
        self.mlp_mmwave = MLP(1024, mlp_hidden_dim, embed_dim, mlp_layers)
        self.mmwave_projection = torch.nn.Linear(embed_dim, embed_dim)

        self.distill_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.step = step
    
    def encode_csi(self, csis):
        x = self.mlp_csi(csis.cuda())
        x = self.shared_mlp(x)
        if self.step == 1 or self.step == 2:
            x = self.csi_projection(x)
        # embed = []
        # for csi in csis:
        #     # x = csi.transpose(0, 1).reshape(1000, 30, 3, 3)[::4, :, :, :1].cuda().float()
        #     # x = x.view(x.size(0), -1).unsqueeze(0).unsqueeze(0)
        #     # x = self.wisppn(x)
        #     x = self.mlp_csi(csi.cuda())
        #     x = self.shared_mlp(x)
        #     if self.step == 1 or self.step == 2:
        #         x = self.csi_projection(x)
        #     averaged_output = x.mean(dim=0)
        #     embed.append(averaged_output)
        # x = torch.stack(embed).cuda()
        return x.squeeze()
    
    def encode_mmwave(self, mmwave):
        # x = self.mmWave_model(mmwave.float())#.cuda())
        x = self.mlp_mmwave(mmwave.cuda())
        x = self.shared_mlp(x)
        if self.step == 1 or self.step == 2:
            x = self.mmwave_projection(x)
        return x.squeeze()

    def get_gradients(self, layer_name):
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if layer_name in name and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        return total_grad_norm ** 0.5
    
    # def forward(self, mmwave):
    #     mmwave_embed = self.encode_mmwave(mmwave)

    #     return {'mmwave_embed': mmwave_embed,
    #             'logit_scale': self.logit_scale.exp()}
    
    def forward(self, csi, mmwave):
        csi_embed = self.encode_csi(csi)
        mmwave_embed = self.encode_mmwave(mmwave)

        return {'csi_embed': csi_embed,
                'mmwave_embed': mmwave_embed,
                'logit_scale': self.logit_scale.exp()}

def check_nan(tensor):
    return torch.isnan(tensor).any()


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
    

