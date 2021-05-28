import random
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
from generator import Generator
from pathlib import Path

from dataloader.dp_corpus import DPCorpus
from dataloader.dp_data_loader import DPDataLoader
from dataloader.daily_dialog_parser import DailyDialogParser

USR_END_TKN = "__eou__"
DATA_ROOT = Path('dataloader/daily_dialog/')
BATCH_SIZE = 1


class Seq2SeqGenerator:
    CHECKPOINT = Path('./checkpoints/generator_checkpoint76.pth.tar')

    VOCAB_SIZE = 8000
    MIN_SEQ_LEN = 5
    MAX_SEQ_LEN = 20
    GEN_EMBEDDING_DIM = 256
    GEN_HIDDEN_DIM = 256

    def __init__(self):
        # create dialogue parser
        parser = DailyDialogParser(DATA_ROOT, DPCorpus.SOS, DPCorpus.EOS, DPCorpus.EOU)
        self.corpus = DPCorpus(parser)

        self.generator = Generator(self.corpus.SOS, self.corpus.EOU, self.VOCAB_SIZE,
                                   self.GEN_HIDDEN_DIM, self.GEN_EMBEDDING_DIM, self.MAX_SEQ_LEN).to(device)
        self.generator.load_state_dict(torch.load(self.CHECKPOINT)['state_dict'])

    def sample(self, data):
        pass


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


if __name__ == "__main__":
    with open(DATA_ROOT / "train.txt") as fp:
        dialogue_lines = fp.readlines()
    with open(DATA_ROOT / "valid.txt") as fp:
        valid_dialogue_lines = fp.readlines()

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[USR_END_TKN], extra_ids=0
    )
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)  #  evaluator
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=5e-5)

    generator = Seq2SeqGenerator()
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
    d_loss, g_loss = [], []
    # EVALUATOR TRAINING
    for iter in tqdm(range(1000000)):
        discriminator.train()
        for d_step in range(D_steps):
            discriminator_optimizer.zero_grad()

            context, reply, disc_instance = dialogue_lines_to_io(
                random.choice(dialogue_lines)
            )
            context, reply, disc_instance = (
                context.to(device),
                reply.to(device),
                disc_instance.to(device),
            )
            with torch.no_grad():
                fake_reply = generator.sample(reply)

            output_real = discriminator(
                input_ids=torch.cat([disc_instance, reply[:, :-1]], dim=-1),
                labels=real_label,
            )
            output_fake = discriminator(
                input_ids=torch.cat([disc_instance, fake_reply], dim=-1),
                labels=fake_label,
            )
            loss = output_real.loss + output_fake.loss
            loss.backward()
            discriminator_optimizer.step()

            d_loss.append(loss.item())

    # EVALUATOR TESTING
    for iter in tqdm():
        pass
