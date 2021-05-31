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


def test_discriminator(gen, discriminator, prt=False):
    discriminator.eval()
    tst_dl = gen.get_dataloader("test")
    real, fake = 0, 0

    real_rewards, fake_rewards = [], []

    for total, (context, reply) in enumerate(tqdm(tst_dl)):
        context_txt = generator.detokenize(
            context.cpu().numpy().squeeze(), remove_eou=False
        )
        reply = reply.to(device)

        # print(context_txt)
        disc_instance = tokenizer(
            f"classify: {context_txt} {gen.corpus.EOU} ", return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            fake_reply_txt, _ = generator.generate(context.t(), reply.t())

        reply_txt = generator.detokenize(reply)
        # print("real reply txt: ", reply_txt)
        reply_t5 = tokenizer(reply_txt, return_tensors="pt").input_ids.to(device)
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
    if prt:
        print(
            f"real: [{real / (total + 1)}] fake: [{fake / (total + 1)}] overall: [{0.5 * (real + fake) / (total + 1)}]"
        )
        print(
            f"Real reward: {np.mean(real_rewards)}, Fake reward: {np.mean(fake_rewards)}"
        )


if __name__ == "__main__":
    with open(DATA_ROOT + "train.txt") as fp:
        dialogue_lines = fp.readlines()
    with open(DATA_ROOT + "validation.txt") as fp:
        valid_dialogue_lines = fp.readlines()

    generator = Seq2SeqGenerator(device, DATA_ROOT)

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[generator.corpus.EOU], extra_ids=0
    )
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(
        device
    )  #  evaluator
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=1e-3)

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
    trn_dl = generator.get_dataloader("train")
    generator.model.eval()

    for epoch in tqdm(range(10)):
        print(epoch)
        discriminator.train()
        for batch_id, (context, reply) in enumerate(tqdm(trn_dl)):
            discriminator_optimizer.zero_grad()
            context_txt = generator.detokenize(
                context.cpu().numpy().squeeze(), remove_eou=False
            )
            # print("context txt: ", context_txt)
            disc_instance = tokenizer(
                f"classify: {context_txt} ", return_tensors="pt"
            ).input_ids.to(device)
            # print("disc instance: ", f"classify: {context_txt} ")
            reply = reply.to(device)

            with torch.no_grad():
                fake_reply, _ = generator.generate(context.t(), reply.t())
                # print("fake reply txt: ", fake_reply)

            reply_txt = generator.detokenize(reply)
            # print("real reply txt: ", reply_txt)
            reply_t5 = tokenizer(reply_txt, return_tensors="pt").input_ids.to(device)
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
                test_discriminator(generator, discriminator, prt=True)
