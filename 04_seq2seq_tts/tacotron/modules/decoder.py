"""Decoders"""

import collections
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.seq2seq import dynamic_decode, BasicDecoder
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as tf_helper
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from modules.layers import *
from modules.modules import *
from modules.ops import *
from modules.decoder_cells import *

class TFCustomDecoderOutput(
        collections.namedtuple("TFCustomDecoderOutput", ("rnn_output",
                                                         "token_output",
                                                         "sample_id"))):
    pass


class TFCustomDecoder(decoder.Decoder):
    """
    Custom sampling decoder.

    Allows for stop token prediction at inference time
    and returns equivalent loss in training time.

    Note:
    Only use this decoder with Tacotron 2 as it only accepts tacotron custom 
    helpers
    """

    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize CustomDecoder.
        Args:
            cell: An `RNNCell` instance.
            helper: A `Helper` instance.
            initial_state: A (possibly nested tuple of...) tensors and 
            TensorArrays. The initial state of the RNNCell.
            output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
                `tf.layers.Dense`. Optional layer to apply to the RNN output 
                prior to storing the result or sampling.
        Raises:
            TypeError: if `cell`, `helper` or `output_layer` have an incorrect 
            type.
        """
        rnn_cell_impl.assert_like_rnncell(type(cell), cell)
        if not isinstance(helper, tf_helper.Helper):
            raise TypeError(
                "helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None
                and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" %
                type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer._compute_output_shape(
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return TFCustomDecoderOutput(
            rnn_output=self._rnn_output_size(),
            token_output=self._helper.token_output_size,
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return TFCustomDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            tf.float32,
            self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
            name: Name scope for any created operations.
        Returns:
            `(finished, first_inputs, initial_state)`.
        """
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """Perform a custom decoding step.
        Enables for dyanmic <stop_token> prediction
        Args:
            time: scalar `int32` tensor.
            inputs: A (structure of) input tensors.
            state: A (structure of) state tensors and TensorArrays.
            name: Name scope for any created operations.
        Returns:
            `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "TFCustomDecoderStep", (time, inputs, state)):
            # Call outputprojection wrapper cell
            (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

            # apply output_layer (if existant)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids,
                stop_token_prediction=stop_token)

        outputs = TFCustomDecoderOutput(cell_outputs, stop_token, sample_ids)
        return (outputs, next_state, next_inputs, finished)



class BasicAttentionDecoder():
    "Base class for decoder with attention"
    def __init__(self,
                 prenet_units=[256,256],
                 prenet_auxiliary_feature=None,
                 output_dim=80,
                 outputs_per_step=2,
                 dropout_rate=0.5,
                 is_training=False,
                 attention_mechanism=None,
                 max_iters=2000):
        super(BasicAttentionDecoder, self).__init__()
        self.prenet_units = prenet_units
        self.prenet_auxiliary_feature = prenet_auxiliary_feature
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.outputs_per_step = outputs_per_step
        self.is_training = is_training
        self.attention_mechanism = attention_mechanism
        self.max_iters = max_iters

    def initialize(self):
        # build init modules
        self.init_modules()
        self.add_custom_module()

    def init_modules(self):
        """get initial modules"""
        self.prenet = PreNet(prenet_units=self.prenet_units,
                             dropout_rate=self.dropout_rate,
                             activation_fn=tf.nn.relu,
                             auxiliary_feature=self.prenet_auxiliary_feature,
                             is_training=self.is_training,
                             scope="decoder_prenet")

        # output projection for getting final feature dimensions
        self.output_projection = OutputProjection(
            units=self.output_dim * self.outputs_per_step,
            scope="linea_transform")

    def add_custom_module(self):
        """Add your custom modules here"""
        raise NotImplementedError(
            "You may need to add your own module (if not, just use 'pass' operator).")

    def get_helper(self, batch_size, hparams, input_length=None,
                   targets=None, stop_token_targets=None, global_step=None):
        if self.is_training:
            helper = TrainingHelper(batch_size,
                                    targets,
                                    stop_token_targets,
                                    hparams,
                                    global_step)
        else:
            helper = TestHelper(batch_size, hparams)
        return helper

    def build_rnn_cell(self):
        """Merge all modules into a cell"""
        return BasicTacoDecoderCell(prenet=self.prenet,
                                    rnn_cell=self.decoder_rnn,
                                    attention_computer=self.attention_computer,
                                    output_projection=self.output_projection,
                                    stop_projection=self.stop_projection,
                                    decoder_rnn_init_state=self.decoder_rnn_init_state,
                                    attention_mechanism=self.attention_mechanism,
                                    name="BasicTacoDecoderCell")


class TacotronDecoder(BasicAttentionDecoder):
    """Tacotron2 Decoder"""
    def __init__(self, 
                 prenet_units=[256,256],
                 decoder_rnn_layers=2,
                 decoder_rnn_units=1024,
                 prenet_auxiliary_feature=None,
                 rnn_auxiliary_feature=None,
                 dropout_rate=0.2,
                 zoneout_rate=0.1,
                 decoder_rnn_init_state=None,
                 output_dim=80,
                 outputs_per_step=2,
                 is_training=False,
                 attention_mechanism=None,
                 max_iters=2000):
        BasicAttentionDecoder.__init__(self,
                                       prenet_units=prenet_units,
                                       prenet_auxiliary_feature=prenet_auxiliary_feature,
                                       output_dim=output_dim,
                                       outputs_per_step=outputs_per_step,
                                       dropout_rate=dropout_rate,
                                       is_training=is_training,
                                       attention_mechanism=attention_mechanism,
                                       max_iters=max_iters)
        self.decoder_rnn_layers = decoder_rnn_layers
        self.decoder_rnn_units = decoder_rnn_units
        self.rnn_auxiliary_feature = rnn_auxiliary_feature
        self.decoder_rnn_init_state = decoder_rnn_init_state
        self.zoneout_rate = zoneout_rate
        self.attention_mechanism = attention_mechanism

    def add_custom_module(self):
        # Decoder RNN
        self.decoder_rnn = DecoderRNN(num_layers=self.decoder_rnn_layers, 
                                      num_units=self.decoder_rnn_units, 
                                      zoneout_rate=self.zoneout_rate,
                                      is_training=self.is_training, 
                                      scope="decoder_lstm")

        # Stop projection to predict stop token
        self.stop_token_projection = StopProjection(
            self.is_training,
            shape=self.outputs_per_step,
            scope='stop_token_projection')

    def build_rnn_cell(self, memory, memory_length):
        return BasicTacoDecoderCell(prenet=self.prenet,
                                    rnn_cell=self.decoder_rnn,
                                    attention_mechanism=self.attention_mechanism,
                                    output_projection=self.output_projection,
                                    stop_token_projection=self.stop_token_projection,
                                    memory=memory,
                                    decoder_rnn_init_state=self.decoder_rnn_init_state,
                                    rnn_auxiliary_feature=self.rnn_auxiliary_feature,
                                    name='TacoLSADecoderCell')


    def __call__(self, memory, memory_length, hparams=None, targets=None,
                 targets_length=None, stop_token_targets=None, global_step=None):
        # Init base modules and custom modules
        self.initialize()
        batch_size = tf.shape(memory)[0]
        # seq2seq helper
        self.helper = self.get_helper(
            batch_size, 
            hparams, 
            targets=targets, 
            stop_token_targets=stop_token_targets,
            global_step=global_step)
        # seq2seqd decoder cell 
        self.decoder_cell = self.build_rnn_cell(memory, memory_length)
        self.decoder_state = self.decoder_cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)
        # dynamic decode
        (outputs, stop_token_prediction, _), \
        final_decoder_state, _ = dynamic_decode(
            TFCustomDecoder(self.decoder_cell, self.helper, self.decoder_state),
            impute_finished=False,
            maximum_iterations=self.max_iters,
            swap_memory=False)
        return outputs, stop_token_prediction, final_decoder_state
