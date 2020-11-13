import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils.util import bit_merge 


class WaveRNNCell(torch.jit.ScriptModule):
    def __init__(self, gru, gru_channels, fc_coarse, fc_fine):
        super().__init__()
        self.gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        self.gru_cell.weight_hh.data = gru.weight_hh_l0.data
        self.gru_cell.weight_ih.data = gru.weight_ih_l0.data
        self.gru_cell.bias_hh.data = gru.bias_hh_l0.data
        self.gru_cell.bias_ih.data = gru.bias_ih_l0.data
        self.gru_channels = gru_channels
        self.split_size = gru_channels // 2
        self.fc_coarse = fc_coarse
        self.fc_fine = fc_fine

    @torch.jit.script_method
    def forward_c(self, inputs, conditions, h):
        h = self.gru_cell(torch.cat([conditions, inputs], dim=1), h)
        h_c, _ = torch.split(h, self.split_size, dim=1)
        return self.fc_coarse(h_c)

    @torch.jit.script_method
    def forward_f(self, inputs, conditions, h):
        h = self.gru_cell(torch.cat([conditions, inputs], dim=1), h)
        _, h_f = torch.split(h, self.split_size, dim=1)
        o_f = self.fc_fine(h_f)
        return (o_f, h)


class WaveRNN(nn.Module):
    def __init__(self,
                 quantization_channels=256,
                 gru_channels=896,
                 fc_channels=896,
                 lc_channels=80):
        super().__init__()
        self.quantization_channels = quantization_channels
        self.gru_channels = gru_channels
        self.split_size = gru_channels // 2
        self.fc_channels = fc_channels
        self.lc_channels = lc_channels
        self.gru = nn.GRU(lc_channels + 3, gru_channels, batch_first=True)

        self.fc_coarse = nn.Sequential(
            nn.Linear(self.split_size, fc_channels), nn.ReLU(),
            nn.Linear(fc_channels, quantization_channels))
        self.fc_fine = nn.Sequential(
            nn.Linear(self.split_size, fc_channels), nn.ReLU(),
            nn.Linear(fc_channels, quantization_channels))

        self.register_buffer('mask', self.create_mask())

    def create_mask(self):
        coarse_mask = torch.cat([
            torch.ones(self.split_size, self.lc_channels + 2),
            torch.zeros(self.split_size, 1)
        ],
                                dim=1)
        i2h_mask = torch.cat(
            [coarse_mask,
             torch.ones(self.split_size, self.lc_channels + 3)],
            dim=0)
        return torch.cat([i2h_mask, i2h_mask, i2h_mask], dim=0)

    def sparse_mask(self):
        pass

    def forward(self, inputs, conditions):
        x = torch.cat([conditions, inputs], dim=-1)
        h, h_n = self.gru(x)

        h_c, h_f = torch.split(h, self.split_size, dim=2)

        o_c = self.fc_coarse(h_c)
        p_c = F.log_softmax(o_c, dim=2)

        o_f = self.fc_fine(h_f)
        p_f = F.log_softmax(o_f, dim=2)

        return p_c, p_f, h_n.squeeze(0)

    def after_update(self):
        with torch.no_grad():
            self.gru.weight_ih_l0.data.mul_(self.mask)

    def to_cell(self):
        return WaveRNNCell(self.gru, self.gru_channels, self.fc_coarse,
                           self.fc_fine)

    def generate(self, conditions):
        rnn_cell = self.to_cell()
        seq_len = conditions.size(1)
        batch_size = conditions.size(0)
        h = torch.zeros(batch_size, self.gru_channels,
                        dtype=conditions.dtype,
                        device=conditions.device)

        c_val = torch.zeros(batch_size, dtype=conditions.dtype,
                            device=conditions.device)
        f_val = torch.zeros(batch_size, dtype=conditions.dtype,
                            device=conditions.device)
        zero = torch.zeros(batch_size, dtype=conditions.dtype,
                           device=conditions.device)
        output = []

        raise NotImplementedError(
            "You must implement the generate logic here.\n"
            "Tips:\n"
            "1. We have created the autoregressive loop and several initial states for you.\n"
            "2. After sampling the coarse and fine parts of a sample, you can use bit_merge to combine the two 8-bit values into the 16-bit pcm sample.\n"
            "3. Attention: The wavernn model take floats as input, so use may need to transform the sampled categories into floats like what we did in training (2 * category.float() / 255 - 1.0) before passing into the next step.\n"
            "4. In order to use GPU for decoding, any time you want to create a new tensor, specify the dtype and device. (for example, line 98) ")

        start = time.time()
        for i in range(seq_len):
            # combine initial states for wavernn

            # sample coarse

            # sample fine

            # combine
            sample = bit_merget(c, f)
            output.append(sample)

            if (i + 1) % 100 == 0:
                speed = int((i + 1) / (time.time() - start))
                print(f'{i+1}/{seq_len} -- Speed: {speed} samples/sec\r')

        return torch.stack(output, dim=1).squeeze()
