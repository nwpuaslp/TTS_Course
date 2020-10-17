# coding=utf-8
""" """
import tensorflow as tf
from  utils import  *


class AcousticModel(object):
    def __init__(self, hparams, features, is_training=False, is_validation=False):
        
        self.labels = features[0]
        self.input_length = features[1]
        self.targets = features[2]

        self.acoustic_dimension = hparams.acoustic_dim
        self.batch_size = tf.shape(self.labels)[0]
        self.hparams = hparams
        self.is_training = is_training
        self.is_validation = is_validation
        
        #############################################################
        # complete your code

        outputs = self.labels
        self._output_module = tf.layers.Dense(self.hparams.acoustic_dim, name="linear_output")
        outputs = self._output_module(outputs)

        #############################################################

        self.outputs = outputs

    def is_black(self, name, blacklist):
        return any([black in name for black in blacklist])

    def l2_regularization_loss(self, weights, scale, blacklist):
        target_weights = [tf.nn.l2_loss(w)
                          for w in weights if not self.is_black(w.name, blacklist)]
        l2_loss = sum(target_weights) * scale
        tf.losses.add_loss(l2_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
        return l2_loss

    def add_loss(self, global_step):

        self.loss = tf.reduce_sum(
            0.5 * tf.square(self.outputs - self.targets), axis=[2])
        # Mask the loss.
        mask = tf.cast(
            tf.sequence_mask(self.input_length, tf.shape(self.outputs)[1]), tf.float32)
        self.loss *= mask
        # Average over actual sequence lengths.
        self.loss = tf.reduce_mean(
            tf.reduce_sum(self.loss, axis=[1]) / tf.cast(self.input_length, tf.float32))


    def learning_rate_exponential_decay(self,
                                        global_step,
                                        staircase):
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(self.hparams.initial_learning_rate,
                                        global_step - self.hparams.start_decay,
                                        self.hparams.decay_steps,
                                        self.hparams.decay_rate,
                                        staircase,
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, self.hparams.final_learning_rate),
                          self.hparams.initial_learning_rate)

    def add_optimizer(self, global_step):
        '''
        Adds optimizer. Sets "gradients" and "optimize" fields.
        add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step
            in training
        '''
        with tf.variable_scope('optimizer') as scope:
            self.learning_rate = self.learning_rate_exponential_decay(
                global_step, staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               self.hparams.adam_beta1,
                                               self.hparams.adam_beta2,
                                               self.hparams.adam_epsilon)

            self.gradients, variables = zip(
                *optimizer.compute_gradients(self.loss))
            clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, self.hparams.gradclip_value)

            with tf.control_dependencies(tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(
                    zip(clipped_gradients, variables),
                    global_step=global_step)

    def add_tensorboard_stats(self):
        # Write loss curve to tensorboard
        with tf.variable_scope('train_stats') as scope:
            tf.summary.scalar('loss', self.loss, collections=["train"])
            
        return tf.summary.merge_all("train")



class DurationModel(object):
    def __init__(self, hparams, features, is_training=False, is_validation=False):
        
        self.labels = features[0]
        self.input_length = features[1]
        self.targets = features[2]
        
        self.dur_dimension = hparams.dur_dim
        self.batch_size = tf.shape(self.labels)[0]
        self.hparams = hparams
        self.is_training = is_training
        self.is_validation = is_validation
        
        #############################################################
        # complete your code

        outputs = self.labels
        self._output_module = tf.layers.Dense(self.hparams.dur_dim, name="linear_output")
        outputs = self._output_module(outputs)
        
        #############################################################

        self.outputs = outputs

    def is_black(self, name, blacklist):
        return any([black in name for black in blacklist])

    def l2_regularization_loss(self, weights, scale, blacklist):
        target_weights = [tf.nn.l2_loss(w)
                          for w in weights if not self.is_black(w.name, blacklist)]
        l2_loss = sum(target_weights) * scale
        tf.losses.add_loss(l2_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
        return l2_loss

    def add_loss(self, global_step):
        
        self.loss = tf.reduce_sum(
            0.5 * tf.square(self.outputs - self.targets), axis=[2])
        # Mask the loss.
        mask = tf.cast(
            tf.sequence_mask(self.input_length, tf.shape(self.outputs)[1]), tf.float32)
        self.loss *= mask
        # Average over actual sequence lengths.
        self.loss = tf.reduce_mean(
            tf.reduce_sum(self.loss, axis=[1]) / tf.cast(self.input_length, tf.float32))


    def learning_rate_exponential_decay(self,
                                        global_step,
                                        staircase):
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(self.hparams.initial_learning_rate,
                                        global_step - self.hparams.start_decay,
                                        self.hparams.decay_steps,
                                        self.hparams.decay_rate,
                                        staircase,
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, self.hparams.final_learning_rate),
                          self.hparams.initial_learning_rate)

    def add_optimizer(self, global_step):
        '''
        Adds optimizer. Sets "gradients" and "optimize" fields.
        add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step
            in training
        '''
        with tf.variable_scope('optimizer') as scope:
            self.learning_rate = self.learning_rate_exponential_decay(
                global_step, staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               self.hparams.adam_beta1,
                                               self.hparams.adam_beta2,
                                               self.hparams.adam_epsilon)

            self.gradients, variables = zip(
                *optimizer.compute_gradients(self.loss))
            clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, self.hparams.gradclip_value)

            with tf.control_dependencies(tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(
                    zip(clipped_gradients, variables),
                    global_step=global_step)

    def add_tensorboard_stats(self):
        # Write loss curve to tensorboard
        with tf.variable_scope('train_stats') as scope:
            tf.summary.scalar('loss', self.loss, collections=["train"])
            
        return tf.summary.merge_all("train")


