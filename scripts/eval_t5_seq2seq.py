import random
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
from src.seq2seq_gen import Seq2SeqGenerator
from pathlib import Path

import torch
import torch.nn  as nn

USR_END_TKN = "__eou__"
DATA_ROOT = 'data/'
BATCH_SIZE = 1
ADV_EVAL_EPOCHS = 10

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("RUNNIG ON CUDA")
else:
    device = torch.device('cpu')
    print("RUNNING ON CPU")


def dialogue_lines_to_io(lines, prt=False, split_point=None):
    lines = lines.rstrip("\n").rstrip(USR_END_TKN)
    lines = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", lines)
    lines = lines.split(USR_END_TKN)
    if split_point is None:
        num_lines = len(lines)
        split_point = np.random.randint(num_lines - 1) + 1
    context, reply = USR_END_TKN.join(lines[:split_point]), lines[split_point]
    if prt:
        print("CONTEXT")
        for i, line in enumerate(lines[:split_point]):
            tab = "\t" if i % 2 == 1 else ""
            print(f"{tab}{line}")
        print("REPLY")
        print(reply)
    context_tok = tokenizer(
        f"reply to: {context}", return_tensors="pt", max_length=512
    ).input_ids
    reply = tokenizer(reply, return_tensors="pt", max_length=512).input_ids
    disc_instance = tokenizer(
        f"classify : {context} {USR_END_TKN} ", return_tensors="pt", max_length=512
    ).input_ids
    return context_tok, reply, disc_instance


def dialogue_lines_to_seq2seq(lines, corpus, prt=False, split_point=None):
    lines = lines.rstrip("\n").rstrip(USR_END_TKN)
    lines = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", lines)
    lines = lines.split(USR_END_TKN)
    if split_point is None:
        num_lines = len(lines)
        split_point = np.random.randint(num_lines - 1) + 1
    context, reply = USR_END_TKN.join(lines[:split_point]), lines[split_point]
    if prt:
        print("CONTEXT")
        for i, line in enumerate(lines[:split_point]):
            tab = "\t" if i % 2 == 1 else ""
            print(f"{tab}{line}")
        print("REPLY")
        print(reply)
    context_tok = torch.LongTensor(corpus.utterance_to_ids(context)).unsqueeze(0).t().long()
    reply_tok = torch.LongTensor(corpus.utterance_to_ids(reply)).unsqueeze(0).t().long()
    disc_instance = tokenizer(
        f"classify : {context} {USR_END_TKN} ", return_tensors="pt", max_length=512
    ).input_ids
    return context_tok, reply_tok, disc_instance


def test_discriminator(gen, disc):
    gen


if __name__ == "__main__":
    with open(DATA_ROOT + "train.txt") as fp:
        dialogue_lines = fp.readlines()
    with open(DATA_ROOT + "validation.txt") as fp:
        valid_dialogue_lines = fp.readlines()

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[USR_END_TKN], extra_ids=0
    )
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)  #  evaluator
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=5e-5)

    generator = Seq2SeqGenerator(device, DATA_ROOT)
    best_loss = np.float("inf")

    real_label, fake_label = (
        tokenizer("real", return_tensors="pt").input_ids.to(device),
        tokenizer("fake", return_tensors="pt").input_ids.to(device),
    )
    """
    Adversarial evaluation scheme is as follows:
        - we first need to train the evaluator to distinguish between the machien generatred and human responses
        - then we test the evaluator on an unsees dataset hoping that it would achieve 50% accuracy 
    """
    rewards = []
    d_loss = []
    # EVALUATOR TRAINING
    discriminator.train()

    for i in tqdm(range(1_000_000)):
        discriminator_optimizer.zero_grad()

        context, reply, disc_instance = dialogue_lines_to_seq2seq(
            random.choice(dialogue_lines), generator.corpus
        )
        context, reply, disc_instance = (
            context.to(device),
            reply.to(device),
            disc_instance.to(device),
        )

        with torch.no_grad():
            fake_reply, _ = generator.sample(context, torch.zeros(20, 1).long())
            fake_reply = torch.LongTensor(tokenizer(fake_reply).input_ids).view(1, -1)

        output_real = discriminator(
            input_ids=torch.cat([disc_instance, reply[:, :-1].view(1, -1)], dim=-1),
            labels=real_label)

        output_fake = discriminator(
            input_ids=torch.cat([disc_instance, fake_reply], dim=-1),
            labels=fake_label)

        loss = output_real.loss + output_fake.loss
        loss.backward()
        discriminator_optimizer.step()
        d_loss.append(loss.item())
        if i + 1 % 10_000 == 0:
            print(f'ADV EVAL train loss: [{loss.item():.5f}]')

    discriminator.eval()
