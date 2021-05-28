import torch

from src.model import Model
from src.constants import GEENRATOR_MAX_LENGTH


class Generator(Model):
    def get_loss(self, *args, **kwargs):
        return self.model(*args, **kwargs).loss

    def generate(self, context, do_sample=False):
        return self.model.generate(
            context, do_sample=do_sample, max_length=GEENRATOR_MAX_LENGTH
        )[:, 1:]

    def get_logprob(self, context, reply):
        return torch.log_softmax(
            self.model(input_ids=context, labels=reply).logits, dim=2
        )[:, range(reply.size(1)), reply[0]]

    def get_prob(self, context, reply):
        return torch.softmax(self.model(input_ids=context, labels=reply).logits, dim=2)[
            :, range(reply.size(1)), reply[0]
        ]

