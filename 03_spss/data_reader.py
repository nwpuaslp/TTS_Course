# coding=utf-8
""" """
import codecs
import os
import queue
import random
import threading

import numpy as np


def read_dur_features(filelist_scpfile, duration_features_dir, labels_dir):
    file_list = []
    with codecs.open(filelist_scpfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            file_id = line
            file_list.append(file_id)
    random.shuffle(file_list)
    cmvn = np.load('cmvn/train_cmvn_dur.npz')

    for file_id in file_list:
        label_path = os.path.join(labels_dir + "/", file_id + '.npy')
        duration_features_path = os.path.join(duration_features_dir, file_id + '.lab')

        label = np.load(label_path)
        duration = np.loadtxt(duration_features_path).astype(np.float32).reshape([-1, 5])
        
        label = (label - cmvn["mean_inputs"].astype(np.float32)) / cmvn["stddev_inputs"].astype(np.float32)
        duration = (duration - cmvn["mean_targets"].astype(np.float32)) / cmvn["stddev_targets"].astype(np.float32)

        input_len = min(label.shape[0], duration.shape[0])
        label = label[:input_len]
        duration = duration[:input_len]

        yield label, input_len, duration


def read_spss_features(filelist_scpfile, acoustic_features_dir, labels_dir):
    file_list = []
    with codecs.open(filelist_scpfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            file_id = line
            file_list.append(file_id)
    random.shuffle(file_list)
    cmvn = np.load('cmvn/train_cmvn_spss.npz')

    for file_id in file_list:
        label_path = os.path.join(labels_dir + "/", file_id + '.npy')
        acoustic_features_path = os.path.join(acoustic_features_dir, file_id + '.cmp')

        label = np.load(label_path)
        acoustic = np.fromfile(acoustic_features_path, dtype=np.float32).reshape([-1, 75])
        
        label = (label - cmvn["mean_inputs"].astype(np.float32)) / cmvn["stddev_inputs"].astype(np.float32)
        acoustic = (acoustic - cmvn["mean_targets"].astype(np.float32)) / cmvn["stddev_targets"].astype(np.float32)

        input_len = min(label.shape[0], acoustic.shape[0])
        label = label[:input_len]
        acoustic = acoustic[:input_len]

        yield label, input_len, acoustic


class DataReader_Acoustic(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 filelist,
                 acoustic_features_dir=None,
                 labels_dir=None,
                 hparams=None,
                 queue_size=128):
        self.coord = coord
        self.filelist = filelist
        self.acoustic_features_dir = acoustic_features_dir
        self.labels_dir = labels_dir
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)
        self.hparams = hparams

    def dequeue_tts(self, num_elements):
        batch = [self.queue.get(block=True) for i in range(num_elements)]
        random.shuffle(batch)

        label = _prepare_numpy([x[0] for x in batch])
        input_length = np.array([x[1] for x in batch], dtype=np.int32)
        acoustic = _prepare_numpy([x[2] for x in batch])
        
        return label, input_length, acoustic

    def thread_tts_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_spss_features(self.filelist,
                                          self.acoustic_features_dir,
                                          self.labels_dir)
            for label, input_len, acoustic in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                self.queue.put((label, input_len, acoustic))

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_tts_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads


class DataReader_Duration(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord,
                 filelist,
                 acoustic_features_dir=None,
                 labels_dir=None,
                 hparams=None,
                 queue_size=128):
        self.coord = coord
        self.filelist = filelist
        self.acoustic_features_dir = acoustic_features_dir
        self.labels_dir = labels_dir
        self.threads = []
        self.queue = queue.Queue(maxsize=queue_size)
        self.hparams = hparams

    def dequeue_tts(self, num_elements):
        batch = [self.queue.get(block=True) for i in range(num_elements)]
        random.shuffle(batch)

        label = _prepare_numpy([x[0] for x in batch])
        input_length = np.array([x[1] for x in batch], dtype=np.int32)
        acoustic = _prepare_numpy([x[2] for x in batch])

        return label, input_length, acoustic

    def thread_tts_main(self):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = read_dur_features(self.filelist,
                                          self.acoustic_features_dir,
                                          self.labels_dir)
            for label, input_len, acoustic in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                self.queue.put((label, input_len, acoustic))

    def start_threads(self, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_tts_main, args=())
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads


def _prepare_numpy(inputs):
    max_len = max(x.shape[0] for x in inputs)
    return np.stack(
        [_pad_numpy(x, max_len) for x in inputs])


def _pad_numpy(t, length):
    return np.pad(
        t, [(0, length - t.shape[0]), (0, 0)],
        mode='constant',
        constant_values=0.)
