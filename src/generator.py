from src.model import Model


class Generator(Model):
    def get_loss(self, *args, **kwargs):
        return self.model(*args, **kwargs).loss

