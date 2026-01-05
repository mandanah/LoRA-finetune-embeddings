import torch


def multiple_negative_ranking_loss(
    anchor_embeddings, positive_embeddings, temperature=0.05
):
    logits = anchor_embeddings @ positive_embeddings.T
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)
