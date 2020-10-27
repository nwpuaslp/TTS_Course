import numpy as np
import tensorflow as tf
from modules.ops import *

def conv1d_bn_drop(inputs, kernel_size, channels, activation_fn=None, 
                   is_training=False, dropout_rate=0.5, scope="conv1d_bn_drop"):
    """1d convolution followed by batch normalization and dropout."""
    with tf.variable_scope(scope):
        outputs = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=None,
            padding='same')
        outputs = tf.layers.batch_normalization(
            outputs, training=is_training)
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return tf.layers.dropout(outputs, rate=dropout_rate,
                                 training=is_training,
                                 name='dropout_{}'.format(scope))


def conv2d(inputs, filters, kernel_size, strides, 
           activation_fn=None, is_training=False, scope="conv2d"):
    """2d convolution followed by batch normalization."""
    with tf.variable_scope(scope):
        outputs = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same')
        outputs = tf.layers.batch_normalization(
            outputs, training=is_training)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs 

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    """LSTM Cell with Zoneoue"""
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., 
                 zoneout_factor_output=0., state_is_tuple=True, name=None):
        '''
        Initializer with possibility to set different zoneout values for
        cell/hidden states.
        '''
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError(
                'One/both provided Zoneout factors are not in [0, 1]')
        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None \
            else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(
                state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(
                new_state, [0, self._cell._num_units], [-1, num_proj])

        # Apply zoneout
        if self.is_training:
            # nn.dropout takes keep_prob (probability to keep activations) not
            # drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(
                new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(
                new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * \
                new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(
            c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state

   
class CustomProjection():
    """
    Custom projection.
    apply_activation_fn: since maybe it is integrated inside the sigmoid_cross_entropy loss function.
    """
    def __init__(self, num_units, apply_activation_fn, activation_fn=None, scope="CustomProjection"):
        self.num_units = num_units
        self.apply_activation_fn = apply_activation_fn
        self.activation_fn = activation_fn
        self.scope = scope
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            outputs = tf.layers.dense(inputs, 
                                      units=self.num_units, 
                                      activation=None, 
                                      name=self.scope)
            if self.apply_activation_fn:
                outputs = self.activation_fn(outputs)
            return outputs
