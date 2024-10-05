# from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import Callback

from watermark.watermark import Watermark


class WatermarkCallback(Callback):
    def __init__(self, start: int, end: int, watermarker: Watermark, gamma: float, logger) -> None:
        super().__init__()
        self.epoch = 0
        self.start = start
        self.end = end
        self.watermarker = watermarker
        self.gamma = gamma
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        if self.start <= self.epoch < self.end:
            new_model_sd = self.watermarker.embed(pl_module, self.gamma)
            pl_module.load_state_dict(new_model_sd)
            self.logger.info(f'Model tattooed at epoch {self.epoch}')


class WatermarkVerifyCallback(Callback):
    def __init__(self, start: int, watermarker: Watermark, logger) -> None:
        super().__init__()
        self.epoch = 0
        self.start = start
        self.watermarker = watermarker
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        if self.epoch > self.start and self.epoch % 10 == 0:
            success = self.watermarker.extract(pl_module)
            self.logger.info('Tattoo {} at epoch {}'.format('present' if success else 'not present', self.epoch))
