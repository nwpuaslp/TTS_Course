import os
import librosa
import random
import numpy as np
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.io import wavfile
from utils.utils import *
from utils import audio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy import signal

class FeatureExtractor():
    """
    Extractor for extracting acoustic features.

    When you want to extract your own acoustic features, there are some steps you need to do:
        1) Add your acoustic type (e.g. Bark) into 'supported_acoustic_type' in hparams.yaml
        2) Add it into FeatureExtractor.supported_extractors
        3) Write your own extractor (e.g. BarkExtractor) which inherits the class BaseAcousticExtractor
           and implement the function 'extract_features'.

    Noted that maybe you need to inherits this class to modify __call__ function to match your special requirement.
    """
    def __init__(self, hparams, args):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.hparams = hparams
        self.acoustic_type = hparams.acoustic_type
        self.wav_dir = args.wav_dir
        assert (self.acoustic_type in hparams.supported_acoustic_type)
        self.supported_extractors = {"Mel": MelExtractor}
        self.extractor = self.supported_extractors[self.acoustic_type](hparams, args)

    def __call__(self, label_dict):
        acoustic_metadata = self.extractor(label_dict, self.wav_dir)
        total_frames = sum([int(m[3]) for m in acoustic_metadata])
        total_samples = sum([int(m[2]) for m in acoustic_metadata])
        sr = self.hparams.sample_rate
        hours = total_samples / sr / 3600.0

        print("Successfully extract {} utterances, about {:.2f} hours."
              .format(len(acoustic_metadata), hours))
        print('Max acoustic frames length: {}'.format(
            max(int(m[3]) for m in acoustic_metadata)))

        random.shuffle(acoustic_metadata)
        acoustic_dict = {}
        for single_feature in acoustic_metadata:
            feature_index = os.path.splitext(os.path.basename(single_feature[0]))[0]
            acoustic_dict[feature_index] = single_feature

        return acoustic_dict

class BaseAcousticExtractor():
    """
    Base class of acoustic extractor.

    For each acoustic feature extractor, you have to implement the fuction 'extract_features',
    and if you have some special requirements, you may need to reload the __call__ function.

    Noted that the return value (also the orders) is very important for later feature parser.
    """
    def __init__(self, hparams, args):
        super(BaseAcousticExtractor, self).__init__()
        self.hparams = hparams
        self.args = args
        self.base_out_dir = args.out_feature_dir
        os.makedirs(self.base_out_dir, exist_ok=True)
        self.executor = ProcessPoolExecutor(max_workers=args.n_jobs)

    def extract_features(self, label_dict, wav_dir):
        raise NotImplementedError(
            "You need to implement extract_features function based on your feature.")

    def __call__(self, label_dict, wav_dir):
        print('Start to extract {} to {}.'
              .format(self.hparams.acoustic_type, self.out_dir))
        # Use label dict other than directly wav list, because we may
        # not use all wav data.
        acoustic_metadata = self.extract_features(label_dict, wav_dir)
        return acoustic_metadata


class MelExtractor(BaseAcousticExtractor):
    """
    Mel-spectrogram extractor.

    A spectific mel feature extractor for extracting mel-spectrogram, and it also acts as
    an example for your own extractor.
    """
    def __init__(self, hparams, args):
        BaseAcousticExtractor.__init__(self, hparams, args)
        self.out_dir = os.path.join(self.base_out_dir, 'mels')
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_wav_dir = os.path.join(self.base_out_dir, 'aligned_wavs')
        os.makedirs(self.out_wav_dir, exist_ok=True)
        self.acoustic_description = ["wav_filename", "acoustic_filename", "n_samples", "n_frames"]

    def extract_features(self, label_dict, wav_dir):
        futures = []
        for key in label_dict:
            wav_path = os.path.join(wav_dir, key + '.wav')
            out_wav_path = os.path.join(self.out_wav_dir, key + '.wav')
            futures.append(self.executor.submit(partial(extract_mel, wav_path, out_wav_path,
                self.out_dir, key, self.hparams, self.args)))
        return [future.result() for future in tqdm(futures) if future.result()]


def extract_mel(wav_filename, out_wav_path, out_dir, key, hparams, args):
    if not os.path.exists(wav_filename):
        print("Wav file {} doesn't exists.".format(wav_filename))
        return None

    wav = audio.load_wav(wav_filename,
                         sr=hparams.sample_rate)
    # Process wav samples
    wav = audio.trim_silence(wav, hparams)
    n_samples = len(wav)

    # Extract mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    n_frames = mel_spectrogram.shape[1]
    if n_frames > hparams.max_acoustic_length:
        print("Ignore wav {} because the frame number {} is too long (Max {} frames in hparams.yaml)."
              .format(wav_filename, n_frames, hparams.max_acoustic_length))
        return None

    # Align features
    desired_frames = int(min(n_samples / hparams.hop_size, n_frames))
    wav = wav[:desired_frames * hparams.hop_size]
    mel_spectrogram = mel_spectrogram[:, :desired_frames]
    n_samples = wav.shape[0]
    n_frames = mel_spectrogram.shape[1]
    assert(n_samples / hparams.hop_size == n_frames)

    # Save intermediate acoustic features
    mel_filename = os.path.join(out_dir, key + '.npy')
    np.save(mel_filename, mel_spectrogram.T, allow_pickle=False)
    audio.save_wav(wav, out_wav_path, hparams)

    return (wav_filename, mel_filename, n_samples, n_frames)
