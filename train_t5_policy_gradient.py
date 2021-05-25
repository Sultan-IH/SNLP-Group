import random
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

USR_END_TKN = "__eou__"


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
    with open("/home/piotr/nlp/data/train/dialogues_train.txt") as fp:
        dialogue_lines = fp.readlines()
    with open("/home/piotr/nlp/data/validation/dialogues_validation.txt") as fp:
        valid_dialogue_lines = fp.readlines()

    # dialogue_lines = dialogue_lines[:100]

    device = torch.device("cuda")
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[USR_END_TKN], extra_ids=0
    )
    generator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    generator_optimizer = AdamW(generator.parameters(), lr=5e-5)
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=5e-5)

    generator.load_state_dict(torch.load("/home/piotr/nlp/SNLP-Group/generator.pt"))
    discriminator.load_state_dict(
        torch.load("/home/piotr/nlp/SNLP-Group/discriminator.pt")
    )

    best_loss = np.float("inf")

    real_label, fake_label = (
        tokenizer("real", return_tensors="pt").input_ids.to(device),
        tokenizer("fake", return_tensors="pt").input_ids.to(device),
    )

    D_steps, G_steps = 1, 1

    rewards = []
    d_loss, g_loss = [], []
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
                fake_reply = generator.generate(context, do_sample=True, max_length=50)[
                    :, 1:-1
                ]
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

        discriminator.eval()
        for g_step in range(G_steps):
            generator_optimizer.zero_grad()
            context, reply, disc_instance = dialogue_lines_to_io(
                random.choice(dialogue_lines)
            )
            pad_reply = torch.cat([torch.tensor([[0]]), reply], dim=1)
            context, reply, disc_instance, pad_reply = (
                context.to(device),
                reply.to(device),
                disc_instance.to(device),
                pad_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True, max_length=50)
            fake_logits = (
                generator(input_ids=context, decoder_input_ids=fake_reply[:, :-2])
                .logits.max(dim=2)
                .values
            )
            fake_reply = fake_reply[:, 1:-1]

            real_logits = (
                generator(input_ids=context, decoder_input_ids=pad_reply[:, :-2])
                .logits.max(dim=2)
                .values
            )

            with torch.no_grad():
                reward = torch.softmax(
                    discriminator(
                        input_ids=torch.cat([disc_instance, fake_reply], dim=-1),
                        decoder_input_ids=torch.tensor([[0]], device=device),
                    ).logits[0, 0, [490, 9901]],
                    0,
                )[0].item()

            rewards.append(reward)

            loss = -(reward - np.mean(rewards)) * torch.sum(fake_logits) - (
                1 - np.mean(rewards)
            ) * torch.sum(real_logits)
            loss.backward()
            generator_optimizer.step()

            g_loss.append(loss.item())

        n = 10000
        if iter % n == 0:
            print(
                f"Iter: {iter}, Reward: {np.mean(rewards[-n:])}, D-Loss: {np.mean(d_loss[-n:])}, G-Loss: {np.mean(g_loss[-n:])}"
            )

