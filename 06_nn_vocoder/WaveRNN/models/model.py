import numpy as np
import torch
import torch.nn as nn
from layers.wavernn import WaveRNN
from layers.modules import ConvInUpsampleNetwork


class Model(nn.Module):
    def __init__(self,
                 quantization_channels=256,
                 gru_channels=896,
                 fc_channels=896,
                 lc_channels=80,
                 upsample_factor=(5, 5, 8),
                 use_gru_in_upsample=True):
        super().__init__()

        self.upsample = ConvInUpsampleNetwork(upsample_scales=upsample_factor,
                                              upsample_activation="none",
                                              upsample_activation_params={},
                                              mode="nearest",
                                              cin_channels=lc_channels,
                                              use_gru=use_gru_in_upsample)
 
        self.wavernn = WaveRNN(quantization_channels, gru_channels,
                               fc_channels, lc_channels)

    def forward(self, inputs, conditions):
        conditions = self.upsample(conditions.transpose(1, 2))
        return self.wavernn(inputs, conditions[:, 1:, :])

    def after_update(self):
        self.wavernn.after_update()

    def generate(self, conditions):
        self.eval()
        with torch.no_grad():
            conditions = self.upsample(conditions.transpose(1, 2))
            output = self.wavernn.generate(conditions)
        self.train()
        return output
