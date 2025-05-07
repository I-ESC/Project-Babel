import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class Attn(nn.Module):
    def __init__(self, guide):
        super().__init__()

        self.reduce_d1 = nn.Linear(128, 1280)

        self.reduce_d2 = nn.Linear(128, 1280)

        self.weight = nn.Sequential(

            # nn.Linear(2560, 1280),
            # nn.BatchNorm1d(1280),
            # nn.Tanh(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            )

        self.guide_flag = guide

    def forward(self, hidden_state_1, hidden_state_2):
        # print(hidden_state_1.shape, hidden_state_2.shape) # torch.Size([16, 512]) torch.Size([16, 512])
        concat_feature = torch.cat((hidden_state_1, hidden_state_2), dim=1) #[bsz, 512*2]
        activation = self.weight(concat_feature)#[bsz, 2]

        score = F.softmax(activation, dim=1)

        new_score = score

        attn_feature_1 = hidden_state_1 * (new_score[:, 0].view(-1, 1, 1)) 
        attn_feature_2 = hidden_state_2 * (new_score[:, 1].view(-1, 1, 1))

        fused_feature = torch.cat( (attn_feature_1, attn_feature_2), dim=2)
        # print(fused_feature.shape) # torch.Size([16, 16, 1024])
        return fused_feature, new_score[:, 0], new_score[:, 1]


class LinearClassifierAttn(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes, guide):
        super(LinearClassifierAttn, self).__init__()

        self.attn = Attn(guide)

        self.gru = nn.GRU(1024, 120, 2, batch_first=True)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, feature1, feature2):

        # feature1 = feature1.view(feature1.size(0), 16, -1)
        # feature2 = feature2.view(feature2.size(0), 16, -1)

        fused_feature, weight1, weight2 = self.attn(feature1, feature2)
        fused_feature = self.dropout(fused_feature)
        # self.gru = nn.GRU(fused_feature.size(2), 120, 2, batch_first=True).cuda()
        fused_feature, _ = self.gru(fused_feature)
        # print(fused_feature.shape) # torch.Size([16, 16, 120])

        fused_feature = fused_feature.contiguous().view(fused_feature.size(0), -1)
        # print(fused_feature.shape) # torch.Size([16, 1920])
        output = self.classifier(fused_feature)
        # print(output.shape) # torch.Size([16, 27])

        return output, weight1, weight2

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_features=2, weights=None):
        super().__init__()
        self.num_features = num_features
        self.weights = weights if weights is not None else [1] * num_features
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, *features):
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, but got {len(features)}")
        
        normalized_weighted_features = [F.normalize(feat, dim=-1, p=2) * weight for feat, weight in zip(features, self.weights)]
        concat_feature = torch.cat(normalized_weighted_features, dim=1)
        return self.mlp(concat_feature)


