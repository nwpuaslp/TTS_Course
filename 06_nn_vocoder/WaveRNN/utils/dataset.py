import numpy as np
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset
from utils.util import bit_division


class WaveRNNDataset(Dataset):
    def __init__(self, path):
        self.metadata = self._get_metadata(path)

    def __getitem__(self, index):
        # We have assert that len(sample) / hparams.hop_size == len(mel)
        # when extracting acoustic features.
        sample, _ = sf.read(self.metadata[index][1], dtype=np.int16)
        mel = np.load(self.metadata[index][0])
        return sample, mel

    def __len__(self):
        return len(self.metadata)

    def _get_metadata(self, path):
        metadata = []
        with open(os.path.join(path, 'train_scp'), 'r') as f:
            for line in f:
                mel_path = line.strip().replace(
                    'labels', 'acoustic_features/mels').replace('lab', 'npy')
                wav_path = line.strip().replace(
                    'labels',
                    'acoustic_features/aligned_wavs').replace('lab', 'wav')
                metadata.append((mel_path, wav_path))

        return metadata


class WaveRNNCollate(object):
    def __init__(self, upsample_factor, condition_window):
        self.upsample_factor = upsample_factor
        self.condition_window = condition_window

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _collate_fn(self, batch):
        c_batch = []
        f_batch = []
        condition_batch = []
        for sample, mel in batch:
            max_offset = len(mel) - self.condition_window
            c_offset = np.random.randint(0, max_offset)
            s_offset = c_offset * self.upsample_factor
            c, f = bit_division(
                sample[s_offset:s_offset +
                       self.condition_window * self.upsample_factor])
            c_batch.append(c)
            f_batch.append(f)
            condition_batch.append(mel[c_offset:c_offset +
                                       self.condition_window])
        c_batch = torch.LongTensor(np.stack(c_batch))
        f_batch = torch.LongTensor(np.stack(f_batch))
        condition_batch = torch.FloatTensor(np.stack(condition_batch))

        return c_batch, f_batch, condition_batch
