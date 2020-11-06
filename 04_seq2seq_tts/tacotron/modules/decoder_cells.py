import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import check_ops, array_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest
import collections
from modules.layers import *
from modules.ops import *
from modules.modules import *
from modules.attention import *


class BasicTacoDecoderCellState(
    collections.namedtuple("BasicTacoDecoderCellState",
                          ("rnn_cell_state", "attention", "time", "alignments",
                           "alignment_history"))):
    def replace(self, **kwargs):
        return super(BasicTacoDecoderCellState, self)._replace(**kwargs)


class BasicTacoDecoderCell(RNNCell):
    def __init__(self, 
                 prenet, 
                 rnn_cell, 
                 attention_mechanism, 
                 output_projection,
                 stop_token_projection,
                 memory=None,
                 decoder_rnn_init_state=None,
                 rnn_auxiliary_feature=None,
                 name="TacoBasicDecoderCell"):
        super(BasicTacoDecoderCell, self).__init__()
        self.prenet = prenet
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism
        self.output_projection = output_projection
        self.stop_token_projection = stop_token_projection
        if attention_mechanism is None and memory is None:
            raise ValueError(
            "attention_mechanism and memory cannot be all None.")
        if attention_mechanism is not None:
            self.memory = attention_mechanism.values
        else:
            self.memory = memory
        self.batch_size = tf.shape(memory)[0]
        self.decoder_rnn_init_state = decoder_rnn_init_state
        self.rnn_auxiliary_feature= rnn_auxiliary_feature
        self.attention_layer_size = self.memory.get_shape()[-1].value
        self.name_scope = name
        self.encoder_length = tf.shape(self.memory)[1]

        self.get_init_cell_state()
        self.attention_computer = self.get_attention_computer()

    def get_attention_computer(self):
        # Attention computer to calculate alignment and context vector
        return BasicAttentionComputer(
            attention_mechanism=self.attention_mechanism,
            attention_layer=None,
            attention_mask=None)
        

    def get_init_cell_state(self):
        with tf.name_scope(self.name_scope, "CellInit"):
            if self.decoder_rnn_init_state is None:
                self.decoder_rnn_init_state = None
            else:
                final_state_tensor = nest.flatten(self.decoder_rnn_init_state)[-1]
                state_batch_size = (final_state_tensor.shape[0].value
                                    or tf.shape(final_state_tensor)[0])
                with ops.control_dependencies(self.check_batch_size(state_batch_size)):
                    self.decoder_rnn_init_state = nest.map_structure(
                        lambda s: array_ops.identity(
                            s, name="check_decoder_rnn_init_state"),
                            self.decoder_rnn_init_state)
                    

    def check_batch_size(self, batch_size):
        error_message = "For zero_state of {}, Non-matching batch sizes between the memory \
                         and requested batch size.".format(self._base_name)
        return [check_ops.assert_equal(batch_size,
                                       self.batch_size,
                                       message=error_message)]

    @property
    def output_size(self):
        return self.output_projection.units

    @property
    def state_size(self):
        return BasicTacoDecoderCellState(
            rnn_cell_state=self.rnn_cell._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self.attention_layer_size,
            alignments=self.attention_mechanism.alignments_size,
            alignment_history=())

    def zero_state(self, batch_size, dtype):
        assert (self.attention_mechanism is not None)
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self.decoder_rnn_init_state is not None:
                rnn_cell_state = self.decoder_rnn_init_state
            else:
                rnn_cell_state = self.rnn_cell._cell.zero_state(batch_size, dtype)

            with ops.control_dependencies(self.check_batch_size(batch_size)):
                rnn_cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    rnn_cell_state)

            return BasicTacoDecoderCellState(
                rnn_cell_state=rnn_cell_state,
                time=array_ops.zeros([], dtype=tf.int32),
                attention=rnn_cell_impl._zero_state_tensors(
                    self.attention_layer_size, batch_size, dtype),
                alignments=self.attention_mechanism.initial_alignments(
                    batch_size, dtype),
                alignment_history=tensor_array_ops.TensorArray(
                    dtype=dtype,
                    size=0,
                    dynamic_size=True))

    def __call__(self, inputs, state):
        prenet_output = self.prenet(inputs)

        rnn_input = tf.concat([prenet_output, state.attention], axis=-1)
        if self.rnn_auxiliary_feature != None:
            rnn_input = tf.concat([rnn_input, self.rnn_auxiliary_feature], axis=-1)
        rnn_output, next_rnn_cell_state = self.rnn_cell(rnn_input, state.rnn_cell_state)

        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        # compute attention
        context_vector, alignments, cumulated_alignments = \
            self.attention_computer(rnn_output, previous_alignments)

        # projections
        projections_input = tf.concat([rnn_output, context_vector], axis=-1)
        cell_outputs = self.output_projection(projections_input)
        stop_tokens = self.stop_token_projection(projections_input)

        alignment_history = previous_alignment_history.write(
            state.time, alignments)

        next_state = BasicTacoDecoderCellState(
            time=state.time + 1,
            rnn_cell_state=next_rnn_cell_state,
            attention=context_vector,
            alignments=cumulated_alignments,
            alignment_history=alignment_history)

        return (cell_outputs, stop_tokens), next_state

