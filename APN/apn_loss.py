from typing import Dict
import torch
from torch import nn, zeros_like
import torch.nn.functional as F

from APN.apn_consts import CUB_SELECTED_ATTRIBUTES_PER_GROUP, CUB_GROUPS
from APN.apn_utils import add_glasso, get_middle_graph
from APN.protomod import ProtoMod


class ProtoModLoss(nn.Module):
    def __init__(
        self, protomod: ProtoMod, reg_weights: Dict[str, float], use_groups: bool
    ):
        super(ProtoModLoss, self).__init__()

        self.protomod = protomod
        self.reg_weights = reg_weights
        self.use_groups = use_groups

        self.middle_graph = get_middle_graph(protomod.kernel_size)

        # To calculate regularization among groups
        self.groups = CUB_GROUPS
        self.attributes_per_group = CUB_SELECTED_ATTRIBUTES_PER_GROUP

        # Precompute group attribute indices as tensors for faster indexing
        if use_groups:
            self.group_attr_indices = [
                torch.tensor(self.attributes_per_group[group], dtype=torch.long)
                for group in self.groups[:-1]
            ]

    def forward(
        self,
        similarity_scores: torch.Tensor,
        attention_maps: torch.Tensor,
        attribute_labels: torch.Tensor,
    ):
        # L_reg from the APN paper
        attribute_reg_loss = self.reg_weights["attribute_reg"] * F.mse_loss(
            similarity_scores, attribute_labels
        )
        loss = attribute_reg_loss

        batch_size, num_attributes, map_dim, _ = attention_maps.size()

        # L_cpt from the APN paper: Enforces compactness of the attention maps
        peak_id = torch.argmax(
            attention_maps.view(batch_size * num_attributes, -1), dim=1
        )
        peak_mask = self.middle_graph[peak_id, :, :].view(
            batch_size, num_attributes, map_dim, map_dim
        )
        cpt_loss = self.reg_weights["cpt"] * torch.sum(
            F.sigmoid(attention_maps) * peak_mask
        )
        loss += cpt_loss

        prototypes = self.protomod.prototype_vectors.squeeze()
        if self.use_groups:
            # L_AD in the APN paper: Attribute decorrelation loss
            decorrelation_loss = zeros_like(cpt_loss)
            for group_attr_idx in self.group_attr_indices:
                # Only selects prototypes that are relevant for this group - Enforces competition per group
                decorrelation_loss += self.reg_weights["decorrelation"] * add_glasso(
                    prototypes, group_attr_idx
                )
            loss += decorrelation_loss
        else:
            # Enforces competition between attributes
            decorrelation_loss = self.reg_weights["decorrelation"] * prototypes.norm(2)
            loss += decorrelation_loss

        return loss, attribute_reg_loss.item(), cpt_loss.item(), decorrelation_loss.item()
