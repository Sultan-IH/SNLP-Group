import argparse
from collections import deque
from os.path import join as path_join

import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm

from src.generator import Generator
from src.discriminator import Discriminator
from src.dataset import DailyDialogueDataset
from src.utils import print_dialogue


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, required=True,
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-iterations", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--generator-path", type=str, default="generator_pretrained.pt")
    parser.add_argument(
        "--discriminator-path", type=str, default="discriminator_pretrained.pt"
    )
    parser.add_argument(
        "--generator-output-path", type=str, default="generator_policy_gradient.pt"
    )
    parser.add_argument(
        "--discriminator-output-path",
        type=str,
        default="discriminator_policy_gradient.pt",
    )
    parser.add_argument("--discriminator-steps", type=int, default=1)
    parser.add_argument("--generator-steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--teacher-forcing", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")

    generator = Generator.from_file(args.generator_path).to(device)
    discriminator = Discriminator.from_file(
        args.discriminator_path, tokenizer=generator.tokenizer
    ).to(device)
    train_dataset = DailyDialogueDataset(
        path_join(args.dataset_path, "train/dialogues_train.txt"),
        tokenizer=generator.tokenizer,
        debug=args.debug,
    )
    valid_dataset = DailyDialogueDataset(
        path_join(args.dataset_path, "validation/dialogues_validation.txt"),
        tokenizer=generator.tokenizer,
        debug=args.debug,
    )

    print(len(train_dataset), len(valid_dataset))

    generator_optimizer = AdamW(generator.parameters(), lr=args.lr)
    discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.lr)

    rewards = deque([], maxlen=args.log_every * args.generator_steps)
    generator_loss = deque([], maxlen=args.log_every * args.generator_steps)
    discriminator_loss = deque([], maxlen=args.log_every * args.discriminator_steps)
    best_reward = 0

    for iter in tqdm(range(args.num_iterations)):
        generator.eval()
        discriminator.train()
        for _ in range(args.discriminator_steps):
            discriminator_optimizer.zero_grad()
            context, real_reply = train_dataset.sample()
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)
            loss, _, _ = discriminator.get_loss(context, real_reply, fake_reply)
            loss.backward()
            discriminator_optimizer.step()

            discriminator_loss.append(loss.item())

        generator.train()
        discriminator.eval()
        for _ in range(args.generator_steps):
            generator_optimizer.zero_grad()
            context, real_reply = train_dataset.sample()
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)

            logprob_real = generator.get_logprob(context, real_reply)
            logprob_fake = generator.get_logprob(context, fake_reply)
            reward_fake = discriminator.get_reward(context, fake_reply)

            baseline = 0 if len(rewards) == 0 else np.mean(rewards)
            loss = -(reward_fake - baseline) * torch.sum(logprob_fake)

            if args.teacher_forcing:
                loss -= (1 - baseline) * torch.sum(logprob_real)

            loss.backward()
            generator_optimizer.step()

            generator_loss.append(loss.item())
            rewards.append(reward_fake)

        if iter % args.log_every == 0:
            mean_reward = np.mean(list(rewards))

            if args.discriminator_steps > 0:
                print(f"Discriminator Loss {np.mean(list(discriminator_loss))}")
            if args.generator_steps > 0:
                print(f"Generator Loss {np.mean(list(generator_loss))}")
                print(f"Mean reward: {mean_reward}\n")

            context, real_reply = valid_dataset.sample()
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)
            reward_fake = discriminator.get_reward(context, fake_reply)

            print_dialogue(
                context=context,
                real_reply=real_reply,
                fake_reply=fake_reply,
                tokenizer=generator.tokenizer,
            )
            print(f"Reward: {reward_fake}\n")

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(discriminator.state_dict(), args.discriminator_output_path)
                torch.save(generator.state_dict(), args.generator_output_path)
            torch.save(
                discriminator.state_dict(), "all_" + args.discriminator_output_path
            )
            torch.save(generator.state_dict(), "all_" + args.generator_output_path)


if __name__ == "__main__":
    main()
