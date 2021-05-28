from src.model import Model
from src.utils import concat_dialogues


class Discriminator(Model):
    def __init__(self) -> None:
        super().__init__()
        self._label_real = self.tokenizer("real", return_tensors="pt").input_ids
        self._label_fake = self.tokenizer("fake", return_tensors="pt").input_ids

        self.register_buffer("label_real", self._label_real)
        self.register_buffer("label_fake", self._label_fake)

    def get_loss(self, context, real_reply, fake_reply, learning_rate):

        loss_real = self.model(
            input_ids=concat_dialogues(context, real_reply, self.tokenizer),
            labels=self.label_real,
        ).loss
        loss_fake = self.model(
            input_ids=concat_dialogues(context, fake_reply, self.tokenizer),
            labels=self.label_fake,
        ).loss

        return loss_real + loss_fake

