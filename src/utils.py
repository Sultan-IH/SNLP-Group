import torch

from src.constants import USR_END_TKN


def concat_dialogues(part_a, part_b, tokenizer):

    join = (
        tokenizer(USR_END_TKN, return_tensors="pt").input_ids[:, :1].to(part_a.device)
    )
    return torch.cat([part_a, join, part_b], dim=1)


def print_dialogue(context, real_reply, tokenizer, fake_reply=None):
    context, real_reply = tokenizer.decode(context[0]), tokenizer.decode(real_reply[0])

    for i, line in enumerate(context.split(USR_END_TKN)):
        prefix = "PERSON A" if i % 2 == 0 else "PERSON B"
        print(f"{prefix}: {line}")
    print(f"REAL REPLY: {real_reply}")

    if fake_reply is not None:
        fake_reply = tokenizer.decode(fake_reply[0])
        print(f"FAKE REPLY: {fake_reply}")

