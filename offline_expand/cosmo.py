import torch
import torch.nn as nn
import torch.nn.functional as F


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
