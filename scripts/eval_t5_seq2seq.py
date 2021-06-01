import argparse
from os.path import join as path_join

import random
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
from src.s2s.generator import Seq2SeqGenerator
from pathlib import Path

import torch
import torch.nn as nn

DATA_ROOT = "../data/"
BATCH_SIZE = 1
ADV_EVAL_EPOCHS = 10

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("RUNNIG ON CUDA")
else:
    device = torch.device("cpu")
    print("RUNNING ON CPU")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, default="../data/",
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--generator-checkpoint", type=str, default="./checkpoints/generator_checkpoint76.pth.tar")
    parser.add_argument(
        "--output-path", type=str, default="discriminator_pretrained.pt"
    )

    args = parser.parse_args()
    return args


def test_discriminator(generator, discriminator, tokenizer, prt=False):
    discriminator.eval()
    tst_dl = generator.get_dataloader("test")
    real, fake = 0, 0

    real_label, fake_label = (
        tokenizer("real", return_tensors="pt").input_ids.to(device),
        tokenizer("fake", return_tensors="pt").input_ids.to(device),
    )

    real_rewards, fake_rewards = [], []

    for context, reply in tqdm(tst_dl):
        reply = reply.to(device)
        context = context.to(device)

        disc_instance = generator.to_t5_tokens(context, tokenizer, prefix='classify: ')

        with torch.no_grad():
            fake_reply_txt, _ = generator.generate(context.t(), reply.t())

        reply_t5 = generator.to_t5_tokens(reply, tokenizer)
        output_real = discriminator(
            input_ids=torch.cat([disc_instance, reply_t5.view(1, -1)], dim=-1),
            labels=real_label,
        )
        fake_reply_t5 = (
            tokenizer(fake_reply_txt, return_tensors="pt")
                .input_ids.view(1, -1)
                .to(device)
        )
        output_fake = discriminator(
            input_ids=torch.cat([disc_instance, fake_reply_t5], dim=-1),
            labels=fake_label,
        )

        real_reward = torch.softmax(output_real.logits[0, 0, [490, 9901]], dim=0)[
            0
        ].item()
        fake_reward = torch.softmax(output_fake.logits[0, 0, [490, 9901]], dim=0)[
            0
        ].item()

        fake += 1 - (fake_reward > 0.5)
        real += real_reward > 0.5

        fake_rewards.append(fake_reward)
        real_rewards.append(real_reward)

    discriminator.train()
    total = len(tst_dl)
    if prt:
        print(
            f"real: [{real / total}] fake: [{fake / total}] overall: [{0.5 * (real + fake) / total}]"
        )
        print(
            f"Real reward: {np.mean(real_rewards)}, Fake reward: {np.mean(fake_rewards)}"
        )


def main():
    args = parse_args()

    generator = Seq2SeqGenerator(device, args.dataset_path, args.generator_checkpoint)

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[generator.corpus.EOU], extra_ids=0
    )
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(
        device
    )
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=1e-3)

    real_label, fake_label = (
        tokenizer("real", return_tensors="pt").input_ids.to(device),
        tokenizer("fake", return_tensors="pt").input_ids.to(device),
    )

    d_loss = []
    discriminator.train()
    trn_dl = generator.get_dataloader("train")
    generator.model.eval()

    for _ in tqdm(range(args.num_epochs)):
        discriminator.train()
        for batch_id, (context, reply) in enumerate(tqdm(trn_dl)):
            reply = reply.to(device)
            context = context.to(device)

            discriminator_optimizer.zero_grad()

            disc_instance = generator.to_t5_tokens(context, tokenizer, prefix='classify: ')

            with torch.no_grad():
                fake_reply, _ = generator.generate(context.t(), reply.t())

            reply_t5 = generator.to_t5_tokens(reply, tokenizer)
            output_real = discriminator(
                input_ids=torch.cat([disc_instance, reply_t5.view(1, -1)], dim=-1),
                labels=real_label,
            )
            fake_reply_t5 = (
                tokenizer(fake_reply, return_tensors="pt")
                    .input_ids.view(1, -1)
                    .to(device)
            )
            output_fake = discriminator(
                input_ids=torch.cat([disc_instance, fake_reply_t5], dim=-1),
                labels=fake_label,
            )

            loss = output_real.loss + output_fake.loss
            loss.backward()
            discriminator_optimizer.step()
            d_loss.append(loss.item())

            if (batch_id + 1) % 10000 == 0:
                print(f"ADV EVAL train loss: [{loss.item():.5f}]")
                test_discriminator(generator, discriminator, tokenizer, prt=True)


if __name__ == '__main__':
    main()
