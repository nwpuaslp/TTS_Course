
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops, array_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
import collections
from modules.layers import *
from modules.ops import *
from modules.attention import *
from modules.decoder_cells import *
from modules.helpers import *

from tensorflow.contrib.seq2seq import dynamic_decode, BasicDecoder
from tensorflow.contrib.seq2seq.python.ops import decoder

class Classifier():
    """Classfier module:
        Both classifiers are fully-connected networks with one 256 unit hidden layer
        followed by a softmax layer to predict the speaker or augmentation posterior.
    """

    def __init__(self, hidden_dim, out_dim, use_GRL, scope="classifier"):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_GRL = use_GRL
        self.scope = scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            if self.use_GRL:
                inputs = flip_gradient(inputs)
            output = tf.layers.dense(inputs, units=self.hidden_dim,
                                     activation=tf.nn.relu, name='in_layer')
            output = tf.layers.dense(output, units=self.out_dim,
                                     activation=None, name='out_layer')
        return output


class PreNet():
    def __init__(self,
                 prenet_units=[512, 512],
                 dropout_rate=0.1,
                 activation_fn=tf.nn.relu,
                 auxiliary_feature=None,
                 is_training=False,
                 scope="PreNet"):
        self.prenet_units = prenet_units
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.auxiliary_feature = auxiliary_feature
        self.is_training = is_training
        self.scope = scope
        if auxiliary_feature is not None:
            self.auxiliary_projection = tf.layers.Dense(
                prenet_units[-1], activation=tf.nn.softsign)

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            outputs = inputs
            for i, unit in enumerate(self.prenet_units):
                outputs = tf.layers.dense(outputs, units=unit, activation=self.activation_fn,
                                          name='dense_{}'.format(i + 1))
                outputs = tf.layers.dropout(outputs,
                                            rate=self.dropout_rate,
                                            training=True,
                                            name='dropout_{}'.format(i + 1) + self.scope)

            if self.auxiliary_feature is not None:
                outputs += self.auxiliary_projection(self.auxiliary_feature)
        return outputs


class PostNet():
    def __init__(self,
                 postnet_num_layers=5,
                 postnet_kernel_size=[5, ],
                 postnet_channels=512,
                 postnet_dropout_rate=0.0,
                 is_training=False,
                 activation_fn=tf.nn.tanh,
                 scope="postnet"):
        super(PostNet, self).__init__()
        self.postnet_num_layers = postnet_num_layers
        self.postnet_kernel_size = postnet_kernel_size
        self.postnet_channels = postnet_channels
        self.postnet_dropout_rate = postnet_dropout_rate
        self.is_training = is_training
        self.activation_fn = activation_fn
        self.scope = scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.postnet_num_layers - 1):
                x = conv1d_bn_drop(
                    inputs=x,
                    kernel_size=self.postnet_kernel_size,
                    channels=self.postnet_channels,
                    activation_fn=self.activation_fn,
                    is_training=self.is_training,
                    dropout_rate=self.postnet_dropout_rate,
                    scope='conv_layer_{}_'.format(i + 1) + self.scope)
            x = conv1d_bn_drop(
                inputs=x,
                kernel_size=self.postnet_kernel_size,
                channels=self.postnet_channels,
                activation_fn=lambda _: _,
                is_training=self.is_training,
                dropout_rate=self.postnet_dropout_rate,
                scope='conv_layer_{}_'.format(5) + self.scope)
        return x


class DecoderRNN():
    def __init__(self, num_layers=2, num_units=1024, zoneout_rate=0.1, is_training=True, scope="DecoderRNN"):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.zoneout_rate = zoneout_rate
        self.scope = scope
        self.rnn_layers = [ZoneoutLSTMCell(
            num_units, is_training,
            zoneout_factor_cell=zoneout_rate,
            zoneout_factor_output=zoneout_rate,
            name='decoder_LSTM_{}'.format(i + 1))
            for i in range(num_layers)]

        self._cell = tf.contrib.rnn.MultiRNNCell(
            self.rnn_layers, state_is_tuple=True)

    def __call__(self, inputs, states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs, states)


class OutputProjection:
    """
    Projection layer to r * acoustic dimensions
    """

    def __init__(self, units=80, activation=None, scope=None):
        super(OutputProjection, self).__init__()

        self.units = units
        self.activation = activation

        self.scope = 'Linear_projection' if scope is None else scope
        self.dense = tf.layers.Dense(
            units=units, activation=activation, name='projection_{}'.
                format(self.scope))

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = self.dense(inputs)
            return output


class StopProjection():
    """
    Projection to a scalar and through a sigmoid activation
    """

    def __init__(self, is_training, shape=1,
                 activation=tf.nn.sigmoid, scope=None):
        super(StopProjection, self).__init__()
        self.is_training = is_training
        self.shape = shape
        self.activation = activation
        self.scope = 'stop_token_projection' if scope is None else scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     activation=None, name='projection_{}'.
                                     format(self.scope))

            # During training, don't use activation as it is integrated inside
            # the sigmoid_cross_entropy loss function
            if self.is_training:
                return output
            else:
                return self.activation(output)



class Classifier():
    """Classfier module:
        Both classifiers are fully-connected networks with one 256 unit hidden layer
        followed by a softmax layer to predict the speaker or augmentation posterior.
    """

    def __init__(self, hidden_dim, out_dim, use_GRL, scope="classifier"):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_GRL = use_GRL
        self.scope = scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            if self.use_GRL:
                inputs = flip_gradient(inputs)
            output = tf.layers.dense(inputs, units=self.hidden_dim,
                                     activation=tf.nn.relu, name='in_layer')
            output = tf.layers.dense(output, units=self.out_dim,
                                     activation=None, name='out_layer')
        return output


