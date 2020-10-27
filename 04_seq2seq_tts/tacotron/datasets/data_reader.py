import io
import os
import random
import threading
import time
import traceback
import zipfile
import queue
import numpy as np
import tensorflow as tf
from operator import itemgetter
from text.phones_mix import phone_to_id

_pad = 0
_stop_token_pad = 1.
_target_pad = -4.1
max_abs_value = 4.

def read_tts_features(filelist_scpfile, acoustic_features_dir, labels_dir, args):
    file_list = []
    with open(filelist_scpfile, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            file_id = line.strip().split('/')[-1].split('.')[0]
            file_list.append(file_id)
    random.shuffle(file_list)

    for file_id in file_list:
        label_path = os.path.join(labels_dir, file_id + '.lab')
        acoustic_features_path = os.path.join(acoustic_features_dir, 'mels', file_id + '.npy')
        phones = []
        with open(label_path, 'r', encoding = 'utf-8') as f:
            line = f.readline()
            content = line.strip().split('|')[2].split(' ')
            for item in content:
                phones.append(phone_to_id[item])
        phones.append(phone_to_id["~"])

        acoustic_targets = np.load(acoustic_features_path)

        phones = np.asarray(phones, np.int32)
        input_length = len(phones)
        targets_length = len(acoustic_targets)

        stop_token_targets = [0.0] * (targets_length - 1) + [1.0]
        stop_token_targets = np.asarray(stop_token_targets, np.float32)

        yield phones, input_length, acoustic_targets, stop_token_targets, targets_length

class DataReader(object):
    def __init__(self,
                 coord,
                 filelist,
                 args,
                 wave_dir=None,
                 acoustic_features_dir=None,
                 labels_dir=None,
                 hparams=None,
                 queue_size=128):
        self.coord = coord
        self.filelist = filelist
        self.wave_dir = wave_dir
        self.acoustic_features_dir = acoustic_features_dir
        self.labels_dir = labels_dir
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)
        self.hparams = hparams
        self.args = args

    def dequeue_tts(self, num_elements):
        batch = [self.queue.get(block=True) for i in range(num_elements)]

        random.shuffle(batch)
        # phones, tones, seg_tags, prosodies ,input_length,acoustic_targets,stop_token_targets,targets_length,spk_id,ref_audio,ref_audio_len
        phones = _prepare_inputs([x[0] for x in batch])
        input_length = np.asarray([x[1] for x in batch], dtype=np.int32)
        acoustic_targets = _prepare_targets([x[2] for x in batch], self.hparams.outputs_per_step)
        stop_token_targets = _prepare_stop_token_targets([x[3] for x in batch], self.hparams.outputs_per_step)
        targets_length = np.asarray([x[4] for x in batch], dtype=np.int32)
        return phones,input_length, acoustic_targets, stop_token_targets, targets_length

    def thread_tts_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_tts_features(self.filelist,
                                         self.acoustic_features_dir,
                                         self.labels_dir, self.args)
            for phones, input_length, acoustic_targets, stop_token_targets, targets_length in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                phones = np.squeeze(phones)

                self.queue.put((phones, input_length, acoustic_targets, stop_token_targets,
                                targets_length))

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_tts_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads


def _prepare_inputs(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets))
    return np.stack(
        [_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _prepare_stop_token_targets(targets, alignment):
    max_len = max([len(t) for t in targets])
    return np.stack([
        _pad_stop_token_target(t, _round_up(max_len, alignment))
        for t in targets
    ])


def _pad_input(x, length):
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(
        t, [(0, length - t.shape[0]), (0, 0)],
        mode='constant',
        constant_values=_target_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder


def _pad_stop_token_target(t, length):
    return np.pad(
        t, (0, length - t.shape[0]),
        mode='constant',
        constant_values=_stop_token_pad)
