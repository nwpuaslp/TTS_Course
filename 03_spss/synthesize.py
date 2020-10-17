# coding=utf-8

""" """
import argparse
import collections
import glob
import os
import time
import copy

import numpy as np
from tqdm import tqdm

from hparam import hparams
from model import *
from utils import *
from scipy import signal


class Synthesizer_duration(object):
    def __init__(self, args):
        self.test_sample = self.get_test_sample()
        self.args = args
        self.make_test_feature()
        self.load_model()
        if not os.path.exists(os.path.join(self.args.output_dir)):
            os.mkdir(os.path.join(self.args.output_dir))

    def load_model(self):
        print('Constructing model ... ...')
        with tf.variable_scope('model') as scope:
            self.model = DurationModel(hparams, self.test_feature, is_validation=True)

            self.outputs = tf.identity(
                self.model.outputs, name="Predict")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, self.args.checkpoint)

    def get_test_sample(self):
        test_sample = collections.namedtuple(
            "Test_sample",
            "labels input_length targets")
        return test_sample

    def make_test_feature(self):
        label = tf.placeholder(tf.float32, [1, None, hparams.dur_label_dim], 'labels')
        input_length = tf.placeholder(tf.int32, [1], 'input_length')
        target = tf.placeholder(tf.float32, [1, None, hparams.dur_dim], 'targets')
        self.test_feature = self.test_sample(
            labels=label,
            input_length=input_length,
            targets=target)

    def __call__(self, label_filename):
        file_id = os.path.splitext(os.path.basename(label_filename))[0]
        labels = np.load(label_filename)
        onehot_label = copy.deepcopy(labels)
        input_length = labels.shape[0]
        cmvn = np.load('cmvn/train_cmvn_dur.npz')
        labels = (labels - cmvn["mean_inputs"].astype(np.float32)) / cmvn["stddev_inputs"].astype(np.float32)
        
        feed_dict = {}
        feed_dict[self.model.input_length] = np.asarray([input_length], dtype=np.int32)
        feed_dict[self.model.labels] = [np.asarray(labels, dtype=np.float32)]
        generated_dur = self.session.run(
            self.outputs, feed_dict=feed_dict)
        generated_dur = generated_dur.reshape(-1, hparams.dur_dim)

        
        cmvn = np.load('cmvn/train_cmvn_dur.npz')
        generated_dur = generated_dur * cmvn["stddev_targets"].astype(np.float32) + cmvn["mean_targets"].astype(np.float32)
        generated_dur = generated_dur.astype(np.int32).astype(np.float32)
        # onehot_label = labels
        acoustic_labels = pending_state_info(onehot_label, generated_dur)

        np.save(self.args.output_dir + '/' + str(file_id), acoustic_labels)
        return acoustic_labels, self.args.output_dir


class Synthesizer_spss(object):
    def __init__(self, args):
        self.test_sample = self.get_test_sample()
        self.args = args
        self.make_test_feature()
        self.load_model()
        if not os.path.exists(os.path.join(self.args.output_dir)):
            os.mkdir(os.path.join(self.args.output_dir))

    def load_model(self):
        print('Constructing model ... ...')
        with tf.variable_scope('model') as scope:
            self.model = AcousticModel(hparams, self.test_feature, is_validation=True)

            self.outputs = tf.identity(
                self.model.outputs, name="Predict")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, self.args.checkpoint)

    def get_test_sample(self):
        test_sample = collections.namedtuple(
            "Test_sample",
            "labels input_length targets")
        return test_sample

    def make_test_feature(self):
        label = tf.placeholder(tf.float32, [1, None, hparams.acoustic_label_dim], 'labels')
        input_length = tf.placeholder(tf.int32, [1], 'input_length')
        target = tf.placeholder(tf.float32, [1, None, hparams.acoustic_dim], 'targets')
        self.test_feature = self.test_sample(
            labels=label,
            input_length=input_length,
            targets=target)

    def __call__(self, label_filename):
        file_id = os.path.splitext(os.path.basename(label_filename))[0]
        labels = np.load(label_filename)
        input_length = labels.shape[0]
        cmvn = np.load('cmvn/train_cmvn_spss.npz')
        labels = (labels - cmvn["mean_inputs"].astype(np.float32)) / cmvn["stddev_inputs"].astype(np.float32)
        
        feed_dict = {}
        feed_dict[self.model.input_length] = np.asarray([input_length], dtype=np.int32)
        feed_dict[self.model.labels] = [np.asarray(labels, dtype=np.float32)]
        generated_acoustic = self.session.run(
            self.outputs, feed_dict=feed_dict)
        generated_acoustic = generated_acoustic.reshape(-1, hparams.acoustic_dim)

        cmvn = np.load('cmvn/train_cmvn_spss.npz')
        generated_acoustic = generated_acoustic * cmvn["stddev_targets"].astype(np.float32) + cmvn["mean_targets"].astype(np.float32)
        cmp_output_path = os.path.join(
            self.args.output_dir, "cmp", '{}.cmp'.format(file_id))
        write_binary_file(generated_acoustic, cmp_output_path)
        cmp_mat = generated_acoustic
        
        mgc = signal.convolve2d(
            cmp_mat[:, : 60], [[1.0 / 3], [1.0 / 3], [1.0 / 3]], mode="same", boundary="symm")
        vuv = cmp_mat[:, 60]
        lf0 = cmp_mat[:, 65]
        bap = cmp_mat[:, 70:]
        inf_float = -1.0e+10
        lf0[vuv < 0.5] = inf_float
        
        write_binary_file(mgc, os.path.join(self.args.output_dir, "mgc", file_id + ".mgc"))
        write_binary_file(lf0, os.path.join(self.args.output_dir, "lf0", file_id + ".lf0"))
        write_binary_file(bap, os.path.join(self.args.output_dir, "bap", file_id + ".bap"))
        
        return generated_acoustic, self.args.output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='',
                        help='Path to model checkpoint')
    parser.add_argument('--label_dir',
                        required=True,
                        help='Path of parametric or end-to-end labels')
    parser.add_argument('--output_dir',
                        required=True,
                        help='folder to contain synthesized acoustic spectrograms')
    parser.add_argument('--model_type',
                        default="AcousticModel")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if(args.model_type == "AcousticModel"):
        synthesizer = Synthesizer_spss(args)
    elif(args.model_type == "DurationModel"):
        synthesizer = Synthesizer_duration(args)
    else:
        raise 'model type error ! '

    print('Starting synthesis')
    labels = [fp for fp in glob.glob(
        os.path.join(args.label_dir, '*'))]
    for i, label_filename in enumerate(tqdm(labels)):
        start = time.time()

        generated_acoustic, acoustic_filename = synthesizer(label_filename)
        end = time.time()
        spent = end - start
        print("Label: {} ".format(
            os.path.basename(label_filename)))


if __name__ == '__main__':
    main()
