from dataclasses import dataclass
import torch


@dataclass
class PairBatch:
    anchor_input_ids: torch.Tensor
    anchor_attention_mask: torch.Tensor
    positive_attention_mask: torch.Tensor
    positive_input_ids: torch.Tensor
