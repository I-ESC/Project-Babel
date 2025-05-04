# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np
def FeatureConstructor(f1, f2, num_positive):

    fusion_weight = np.arange(1, num_positive + 1) / 10#(0.1, 0,2, ..., 0.9)

    fused_feature = []

    for fuse_id in range(num_positive):
        temp_fuse = fusion_weight[fuse_id] * f1 + (1 - fusion_weight[fuse_id]) * f2
        fused_feature.append(temp_fuse)
    
    fused_feature = torch.stack(fused_feature, dim = 1)

    return fused_feature


## contrastive fusion loss with SupCon format: https://arxiv.org/pdf/2004.11362.pdf
class ConFusionLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConFusionLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        contrast_feature = F.normalize(contrast_feature, dim = 1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index
        # print(mask.shape)#[1151, 1152] (btz*9)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )#dig to 0, others to 1 (negative samples)

        mask = mask * logits_mask#positive samples except itself

        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.con_fusion_loss = ConFusionLoss()  # Initialize your ConFusionLoss here

    def forward(self, outputs, optimizer, model):
        image_embed = outputs['image_embed']
        skel_embed = outputs['skeleton_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        skel_embed = F.normalize(skel_embed, dim=-1, p=2)  # Normalize skeleton embedding

        # gather features from all GPUs
        image_embed_all, skel_embed_all = \
            utils.all_gather_batch([image_embed, skel_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ skel_embed_all.t()
        logits_per_skel = logit_scale * skel_embed @ image_embed_all.t()

        # Calculate individual losses
        loss_image = F.cross_entropy(logits_per_image, self.labels)
        loss_skel = F.cross_entropy(logits_per_skel, self.labels)

        # Initialize loss weights
        weight_image, weight_skel = 1.0, 1.0

        # Backward pass for image loss to compute gradients
        optimizer.zero_grad()
        loss_image.backward(retain_graph=True)
        grad_image_mag = model.get_gradients('mlp_image')
        
        # Backward pass for skeleton loss to compute gradients
        optimizer.zero_grad()
        loss_skel.backward(retain_graph=True)
        grad_skel_mag = model.get_gradients('mlp_skeleton')

        # Adjust loss weights based on gradient magnitudes
        weight_image = 1 / grad_image_mag
        weight_skel = 1 / grad_skel_mag

        # Normalize weights
        total_weight = weight_image + weight_skel
        weight_image /= total_weight
        weight_skel /= total_weight

        # Combine losses with new weights and perform final backward pass
        combined_loss = weight_image * loss_image + weight_skel * loss_skel
        return {'loss': combined_loss}