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
    context = tokenizer(
        f"reply to: {context}", return_tensors="pt", max_length=512
    ).input_ids
    reply = tokenizer(reply, return_tensors="pt", max_length=512).input_ids
    disc_instance = tokenizer(
        f"classify : {context} {USR_END_TKN} ", return_tensors="pt", max_length=512
    ).input_ids
    return context, reply, disc_instance


if __name__ == "__main__":
    with open("/home/piotr/nlp/data/train/dialogues_train.txt") as fp:
        dialogue_lines = fp.readlines()
    with open("/home/piotr/nlp/data/validation/dialogues_validation.txt") as fp:
        valid_dialogue_lines = fp.readlines()

    # dialogue_lines = dialogue_lines[:100]
    # valid_dialogue_lines = valid_dialogue_lines[:100]

    device = torch.device("cuda")
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small", additional_special_tokens=[USR_END_TKN], extra_ids=0
    )
    generator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    discriminator = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    optimizer = AdamW(discriminator.parameters(), lr=5e-5)

    generator.load_state_dict(torch.load("/home/piotr/nlp/SNLP-Group/t5.pt"))
    discriminator.load_state_dict(torch.load("/home/piotr/nlp/SNLP-Group/t5.pt"))
    generator.eval()

    best_loss = np.float("inf")

    real_label, fake_label = (
        tokenizer("real", return_tensors="pt").input_ids.to(device),
        tokenizer("fake", return_tensors="pt").input_ids.to(device),
    )

    for epoch in tqdm(range(100)):
        train_loss, valid_loss = [], []
        random.shuffle(dialogue_lines)
        discriminator.train()
        for lines in dialogue_lines:
            optimizer.zero_grad()
            context, reply, disc_instance = dialogue_lines_to_io(lines)
            context, reply, disc_instance = (
                context.to(device),
                reply.to(device),
                disc_instance.to(device),
            )
            with torch.no_grad():
                fake_reply = generator.generate(context)[:, 1:-1]

            loss_real = discriminator(
                input_ids=torch.cat([disc_instance, reply], dim=-1), labels=real_label
            ).loss
            loss_fake = discriminator(
                input_ids=torch.cat([disc_instance, fake_reply], dim=-1),
                labels=fake_reply,
            ).loss
            loss = loss_real + loss_fake
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        discriminator.eval()
        for lines in valid_dialogue_lines:
            context, reply, disc_instance = dialogue_lines_to_io(lines)
            context, reply, disc_instance = (
                context.to(device),
                reply.to(device),
                disc_instance.to(device),
            )
            with torch.no_grad():
                fake_reply = generator.generate(context)[:, 1:-1]
                loss_real = discriminator(
                    input_ids=torch.cat([disc_instance, reply], dim=-1),
                    labels=real_label,
                ).loss
                loss_fake = discriminator(
                    input_ids=torch.cat([disc_instance, fake_reply], dim=-1),
                    labels=fake_reply,
                ).loss
            loss = loss_real + loss_fake
            valid_loss.append(loss.item())

        train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Valid Loss: {valid_loss:.2f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(discriminator.state_dict(), "discriminator.pt")

