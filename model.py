import torch.nn as nn
import torch
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType


class EncoderModel(nn.Module):
    """A wrapper around a transformer model to produce mean-pooled, normalized embeddings."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def mean_pooling(self, last_hidden_state, attention_mask):
        # last_hidden_state shape : (batch_size, seq_length, hidden_size)
        # mask shape : (batch_size, seq_length)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (mask * last_hidden_state).sum(1)
        counted = mask.sum(1)
        counted = torch.clamp(counted, min=1e-6)
        mean_pooled = summed / counted
        return mean_pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        normalize = torch.nn.functional.normalize(pooled_output, p=2, dim=-1)
        return normalize


def build_lora_encoder(model_name: str):
    base = AutoModel.from_pretrained(model_name)
    if "distilbert" in model_name:
        target_modules = ["q_lin", "v_lin"]
    else:
        target_modules = ["query", "value"]
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=4,
        lora_alpha=8,  # LoRA update is scaled as Wnew = W + (α / r) · (B @ A)
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )
    lora_model = get_peft_model(base, lora_config)
    return lora_model
