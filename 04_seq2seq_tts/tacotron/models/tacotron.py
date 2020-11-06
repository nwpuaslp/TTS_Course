from models.basic_model import *
from modules.decoder import *
from modules.layers import *
from modules.ops import *
from utils import infolog
from utils import plot
from utils.utils import *
from text.phones_mix import *

log = infolog.log


class TacotronModel(BaseAcousticModel):

    def __init__(self, features, hparams, is_training=False):
        BaseAcousticModel.__init__(self,
                                   features=features,
                                   hparams=hparams,
                                   is_training=is_training)
        self.phones = features[0]
        self.input_length = features[1]

        self.acoustic_dimension = hparams.acoustic_dim

        self.targets = features[2] if is_training  else None
        self.targets_length = features[4]
        self.stop_token_targets = features[3]
        self.batch_size = tf.shape(self.phones)[0]

    def prepare_input(self):
        hparams = self.hparams
        embedding_table = tf.get_variable(
            'phone_embedding', [len(phone_to_id), hparams.phone_embedding_dim], 
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
        embedded_inputs = tf.nn.embedding_lookup(embedding_table, self.phones)

        return embedded_inputs

    def init_model(self, global_step):
        with tf.variable_scope('inference') as scope:
            hparams = self.hparams
            global_step = tf.train.get_global_step()
            
            # Prepare inputs
            text_inputs = self.prepare_input()

            # Encoder prenet
            encoder_prenet = PreNet(is_training=self.is_training, scope="EncoderPreNet")
            prenet_outputs = encoder_prenet(text_inputs)

            # Encoder module 
            encoder = self.build_encoder()
            encoder_outputs = encoder(inputs=prenet_outputs,
                                      input_length=self.input_length)
            self.encoder_outputs = tf.identity(
                encoder_outputs, name='encoder_outputs')

            # Attention module
            attention_mechanism = self.build_attention_mechanism(encoder_outputs)

            # Decoder module
            decoder = self.build_decoder(
                attention_mechanism=attention_mechanism)
            acoustic_output, stop_token_output, final_decoder_state = decoder(
                memory=encoder_outputs,
                memory_length=self.input_length,
                targets=self.targets,
                targets_length=self.targets_length
                if self.is_training else None,
                stop_token_targets=self.stop_token_targets,
                hparams=hparams,
                global_step=global_step)

            self.alignments = tf.transpose(
                final_decoder_state.alignment_history.stack(), [1, 2, 0])

            decoder_output = tf.reshape(
                acoustic_output, [self.batch_size, -1, self.acoustic_dimension])
            self.decoder_output = tf.identity(
                decoder_output, name='decoder_output')
            stop_token_output = tf.reshape(
                stop_token_output, [self.batch_size, -1])
            self.stop_token_output = tf.identity(
                stop_token_output, name='stop_token_output')
            self.stop_token_targets = self.stop_token_targets
            self.stop_token_binary = tf.nn.sigmoid(stop_token_output)

            # Postnet
            postnet = PostNet(is_training=self.is_training, scope='postnet')
            postnet_output = postnet(decoder_output)

            residual_projection = CustomProjection(
                num_units=self.acoustic_dimension,
                apply_activation_fn=False,
                scope='postnet_projection')

            projected_output = residual_projection(postnet_output)
            self.acoustic_outputs = self.decoder_output + projected_output
            self.outputs = self.acoustic_outputs

    def add_loss(self, global_step):
        '''Adds loss to the model '''
        with tf.variable_scope('loss') as scope:
            # Compute loss of predictions before postnet
            self.before_loss = MaskedMSE(self.targets,
                                         self.decoder_output,
                                         self.targets_length,
                                         hparams=self.hparams)
            # Compute loss of predictions after postnet
            self.after_loss = MaskedMSE(self.targets,
                                        self.acoustic_outputs,
                                        self.targets_length,
                                        hparams=self.hparams)

            # Compute <stop_token> loss 
            self.stop_token_loss = MaskedSigmoidCrossEntropy(
                self.stop_token_targets,
                self.stop_token_output,
                self.targets_length,
                hparams=self.hparams)

            self.loss = (self.before_loss + self.after_loss + self.stop_token_loss)

    def add_optimizer(self, global_step):
        def is_frozen(name, frozen_list):
            return any([frozen in name for frozen in frozen_list])
        if self.hparams.retrain_decoder_only:
            frozen_list = ["phone_embedding", "EncoderCBHG", "EncoderPreNet"]
        else:
            frozen_list = []
        self.vars =  [var for var in tf.trainable_variables()  if not is_frozen(var.name, frozen_list)]

        with tf.variable_scope('optimizer') as scope:
            self.learning_rate = self.learning_rate_exponential_decay(
                global_step, staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               self.hparams.adam_beta1,
                                               self.hparams.adam_beta2,
                                               self.hparams.adam_epsilon)

            self.gradients, variables = zip(
                *optimizer.compute_gradients(self.loss,var_list=self.vars))
            clipped_gradients, _ = tf.clip_by_global_norm(
                self.gradients, self.hparams.gradclip_value)

            with tf.control_dependencies(tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(
                    zip(clipped_gradients, variables),
                    global_step=global_step)

    def add_training_stats(self):
        with tf.variable_scope('train_stats') as scope:
            # Write alignments to tensorboard
            if hasattr(self, 'alignments'):
                img_align = tf.expand_dims(
                    tf.transpose(self.alignments, [0, 2, 1]), -1)
                img_align = plot.gray2rgb(img_align, tf.shape(img_align)[0])
                tf.summary.image('train_align', img_align, max_outputs=3,
                                 collections=["train"])

            tf.summary.scalar('train_loss', self.loss, collections=["train"])
            tf.summary.scalar('learning_rate', self.learning_rate, collections=[
                "train"])  # control learning rate decay speed

            return tf.summary.merge_all("train")
