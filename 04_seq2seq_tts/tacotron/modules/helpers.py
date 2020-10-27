"""Definitions of helpers"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
import math
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op

class TrainingHelper(Helper):
    def __init__(self, batch_size, targets, stop_targets,
                 hparams, global_step):
        with tf.name_scope('TrainingHelper'):
            self._batch_size = batch_size
            self._output_dim = hparams.acoustic_dim
            self._reduction_factor = hparams.outputs_per_step
            self.global_step = global_step
            r = self._reduction_factor

            self._targets = targets[:, r-1::r, :]
            self._lengths = tf.tile(
                [tf.shape(self._targets)[1]], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def token_output_size(self):
        return self._reduction_factor

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(
            self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size]) 

    def next_inputs(self, time, outputs, state, sample_ids,
                    stop_token_prediction, name=None):
        with tf.name_scope(name or 'TrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = self._targets[:, time, :]

            return (finished, next_inputs, state)

class TestHelper(Helper):
    def __init__(self, batch_size, hparams, min_iters=10):
        with tf.name_scope('TestHelper'):
            self._batch_size = batch_size
            self._output_dim = hparams.acoustic_dim
            self._threshold = hparams.stop_threshold
            self._reduction_factor = hparams.outputs_per_step
            self.min_iters = min_iters

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def token_output_size(self):
        return self._reduction_factor

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(
            self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size]) 

    def next_inputs(self, time, outputs, state, sample_ids,
                    stop_token_prediction, name=None):
        with tf.name_scope('TestHelper'):
            termination = tf.greater(stop_token_prediction, self._threshold)

            termination = tf.reduce_any(tf.reduce_all(
                termination, axis=0))

            minimum_require = tf.greater(time, self.min_iters)
            finished = tf.logical_and(termination, minimum_require)

            next_inputs = outputs[:, -self._output_dim:]
            next_state = state
            return (finished, next_inputs, next_state)



def _go_frames(batch_size, output_dim):
    go_frames = tf.tile([[0.0]], [batch_size, output_dim], name='go_frame')
    return go_frames


