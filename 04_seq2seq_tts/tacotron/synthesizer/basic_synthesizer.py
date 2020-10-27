import os
import collections
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import audio
from models.tacotron import *

class BaseSynthesizer():
    """
    Basic Synthesizer for all models.
    Noted that you need to modify make_test_feature to match your feature type.
    """

    def __init__(self, hparams, args):
        super(BaseSynthesizer, self).__init__()
        self.args = args
        self.hparams = hparams
        self.target_acoustic_dir = args.target_acoustic_dir
        self.test_sample = self.get_test_sample()
        self.test_feature = self.make_test_feature()
        self.load_model()

    def get_test_sample(self):
        test_sample = collections.namedtuple(
            "Test_sample", self.tuple_value)
        return test_sample

    def make_test_feature(self):
        raise NotImplementedError(
            "You must implement your own make_test_feature function.")

    def load_model(self):
        with tf.variable_scope('model') as scope:
            self.model = TacotronModel(self.test_feature, self.hparams, is_training=False)
            self.model.init_model(None)
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

    def make_feed_dict(self, label_filename=None):
        raise NotImplementedError(
            "You must implement your own make_feed_dict function.")

    def __call__(self, label_filename):
        hparams = self.hparams
        file_id = os.path.splitext(os.path.basename(label_filename))[0]
        feed_dict = self.make_feed_dict(label_filename)

        generated_acoustic = self.session.run(
            self.outputs, feed_dict=feed_dict)
        
        generated_acoustic = generated_acoustic.reshape(-1, hparams.acoustic_dim)

        acoustic_output_path = os.path.join(
            self.args.output_dir, '{}.npy'.format(file_id))
        np.save(acoustic_output_path, generated_acoustic, allow_pickle=False)


        if self.args.use_gl:
            wav = audio.inv_mel_spectrogram(generated_acoustic.T, hparams)

            wav_output_path = os.path.join(
                self.args.output_dir,
                "{}.wav".format(os.path.splitext(os.path.basename(acoustic_output_path))[0]))
            audio.save_wav(wav, wav_output_path, hparams, norm=True)

        return generated_acoustic, acoustic_output_path 

