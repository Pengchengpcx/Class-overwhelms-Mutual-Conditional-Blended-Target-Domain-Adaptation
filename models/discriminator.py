from torch import nn
from torch.nn.utils import spectral_norm
import torch
from typing import Optional
import torch
import torch.nn as nn

class CatDomain(nn.Module):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Parameters:
        - **in_feature** (int): dimension of the input feature
        - **hidden_size** (int): dimension of the hidden features

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, out_dim: int):
        super(CatDomain, self).__init__()
        self.name = 'Categorical_discriminator'
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.layer3(x)
        
        return y

    def get_parameters(self) :
        return [{"params": self.parameters(), "lr_mult": 1.}]
