import torch.nn.functional as F
from torch import nn
import torch

from APN.apn_consts import NUM_ATTRIBUTES


class ProtoMod(nn.Module):
    def __init__(self, channel_dim: int = 2048, kernel_size: int = 8, num_vectors: int = 1):
        super(ProtoMod, self).__init__()
        self.kernel_size = kernel_size

        prototype_shape = [NUM_ATTRIBUTES * num_vectors, channel_dim, 1, 1]
        self.prototype_vectors = nn.Parameter(
            2e-4 * torch.rand(prototype_shape), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        attention_map = F.conv2d(
            input=x, weight=self.prototype_vectors
        )  # [64, num_attributes x num_vectors, H, W]
        similarity_score = F.max_pool2d(
            attention_map, kernel_size=self.kernel_size
        ).view(batch_size, -1)

        return similarity_score, attention_map
