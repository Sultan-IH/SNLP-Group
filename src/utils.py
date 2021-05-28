import torch

from src.constants import USR_END_TKN


def concat_dialogues(part_a, part_b, tokenizer):
    join = tokenizer(USR_END_TKN)[:, :1].to(part_a.device)
    return torch.cat([part_a, join, part_b], dim=1)
