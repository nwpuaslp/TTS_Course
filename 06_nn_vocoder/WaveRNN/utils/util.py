import numpy as np


def bit_division(samples):
   samples = samples.astype(np.int64) + 2**15
   c = (samples // 256).astype(np.int32)
   f = (samples % 256).astype(np.int32)
   return c, f


def bit_merge(c, f):
   samples = (c * 256 + f - 2**15)
   return samples


class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def apply_moving_average(model, ema):
    for name, param in model.named_parameters():
        if name in ema.shadow:
            ema.update(name, param.data)


def register_model_to_ema(model, ema):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)


