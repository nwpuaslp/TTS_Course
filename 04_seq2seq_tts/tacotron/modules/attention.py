"""Attention classes for seq2seq model."""

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import \
    BahdanauAttention, BahdanauMonotonicAttention
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from modules.layers import *
from modules.ops import *
import math
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score
import functools


class BasicAttentionComputer():
    """
    Basic class for computing attention contexts and alignments.
    This is modified from tensorflow implementation.
    """
    def __init__(self, 
                 attention_mechanism, 
                 attention_layer=None, 
                 memory=None,
                 attention_mask=None):
        super(BasicAttentionComputer, self).__init__()
        if attention_mechanism is None and memory is None:
            raise ValueError(
                "attention_mechanism and memory cannot be all none.")
        self.attention_mechanism = attention_mechanism
        self.attention_layer = attention_layer
        self.attention_mask = attention_mask
        if memory is None:
            self.memory = attention_mechanism.values
        else:
            self.memory = memory
        self.batch_size = tf.shape(self.memory)[0]

    def __call__(self, cell_output, attention_state):
        alignments, next_attention_state = self.attention_mechanism(
            cell_output, state=attention_state)
        if self.attention_mask is not None:
            alignments = alignments*self.attention_mask
            alignments = alignments / \
                tf.clip_by_value(math_ops.reduce_sum(
                    alignments, axis=1, keep_dims=True), 1.0E-20, 10.0)
    
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, 1)
        context = math_ops.matmul(expanded_alignments, self.memory)
        context = array_ops.squeeze(context, [1])
    
        if self.attention_layer is not None:
            attention = self.attention_layer(
                array_ops.concat([cell_output, context], 1))
        else:
            attention = context
    
        return attention, alignments, next_attention_state

