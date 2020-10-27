from modules.decoder import *
from modules.layers import *
from modules.ops import *
from utils import infolog
from utils.utils import *
log = infolog.log

class BaseAcousticModel(object):
    """Base class for acoustic model."""

    def __init__(self, features, hparams, is_training=False):
        self.features = features
        self.hparams = hparams
        self.is_training = is_training

    def init_model(self):
        raise NotImplementedError(
            "You must implement init_model function.")

    def prepare_input(self):
        raise NotImplementedError(
            "You must implement prepare_input function.")

    def add_loss(self):
        raise NotImplementedError(
            "You must implement add_loss function.")

    def add_training_stats(self):
        with tf.variable_scope('train_stats') as scope:
            tf.summary.scalar('train_loss', self.loss, collections=["train"])
            tf.summary.scalar('learning_rate', self.learning_rate, collections=[
                "train"])  
            return tf.summary.merge_all("train")

    def l2_regularization_loss(self, weights, scale, blacklist):
        target_weights = [tf.nn.l2_loss(w)
                          for w in weights if not self.is_black(w.name, blacklist)]
        l2_loss = sum(target_weights) * scale
        tf.losses.add_loss(l2_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
        return l2_loss

    def is_black(self, name, blacklist):
        return any([black in name for black in blacklist])

    def learning_rate_exponential_decay(self,
                                        global_step,
                                        staircase):
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(self.hparams.start_lr,
                                        global_step - self.hparams.start_decay,
                                        self.hparams.decay_steps,
                                        self.hparams.decay_rate,
                                        staircase,
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, self.hparams.end_lr),
                          self.hparams.start_lr)

    def build_attention_mechanism(self, memory):
        hparams = self.hparams
        return BahdanauAttention(
            hparams.attention_dim,
            memory=memory,
            memory_sequence_length=self.input_length)

    def build_encoder(self, encoder_auxiliary_feature=None):
        hparams = self.hparams
        raise NotImplementedError("Homework: implement the CBHG encoder.")

    def build_decoder(self,
                    attention_mechanism,
                    prenet_auxiliary_feature=None,
                    decoder_rnn_init_state=None,
                    decoder_rnn_auxiliary_feature=None):
        hparams = self.hparams
        return TacotronDecoder(
                prenet_auxiliary_feature=prenet_auxiliary_feature,
                rnn_auxiliary_feature=decoder_rnn_auxiliary_feature,
                decoder_rnn_init_state=decoder_rnn_init_state,
                output_dim=self.acoustic_dimension,
                outputs_per_step=hparams.outputs_per_step,
                is_training=self.is_training,
                attention_mechanism=attention_mechanism,
                max_iters=5000)

