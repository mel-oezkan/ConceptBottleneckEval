import torch.nn.functional as F
from torch import nn
import torch


class ProtoMod(nn.Module):
    def __init__(self, channel_dim: int = 2048, kernel_size: int = 8):
        super(ProtoMod, self).__init__()
        self.kernel_size = kernel_size

        prototype_shape = [312, channel_dim, 1, 1]
        self.prototype_vectors = nn.Parameter(
            2e-4 * torch.rand(prototype_shape), requires_grad=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attributes):
        batch_size = x.shape[0]

        attention_map = F.conv2d(
            input=x, weight=self.prototype_vectors
        )  # [64, 312, H, W]
        similarity_score = F.max_pool2d(
            attention_map, kernel_size=self.kernel_size
        ).view(batch_size, -1)

        return similarity_score, attention_map
