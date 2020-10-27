
import os
import glob 
import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import audio
from utils.utils import *
from synthesizer.taco_synthesizer import *

class Synthesizer():
    """Main entrance for synthesizer"""

    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.synthesizer = eval(hparams.synthesizer_type)(hparams, args)

    def __call__(self):
        labels = [fp for fp in glob.glob(
            os.path.join(self.args.label_dir, '*'))]
        for i, label_filename in enumerate(tqdm(labels)):
            start = time.time()

            generated_acoustic, acoustic_filename = self.synthesizer(label_filename)
            if generated_acoustic is None:
                print("Ignore {}".format(os.path.basename(label_filename)))
                continue
            end = time.time()
            spent = end - start
            n_frame = generated_acoustic.shape[0]
            audio_lens = n_frame * self.hparams.hop_size / self.hparams.sample_rate
            print("Label: {}, generated wav length: {}, synthesis time: {}, RTF: {}".format(
                os.path.basename(label_filename), audio_lens, spent, spent / audio_lens))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default='',
                        help='Path to model checkpoint')
    parser.add_argument('--label_dir',
                        required=True,
                        help='Path of parametric or end-to-end labels')
    parser.add_argument('--yaml_conf',
                        default='',
                        help='yaml files for configurations.')
    parser.add_argument('--hparams',
                        default='',
                        help='Overrides hyper parameters as a comma-separated '
                        'list of name=value pairs.')
    parser.add_argument('--target_acoustic_dir', 
                        default=None,
                        help='folder to contain ground-truth acoustic spectrograms')
    parser.add_argument('--output_dir', 
                        required=True,
                        help='folder to contain synthesized acoustic spectrograms')
    parser.add_argument('--alignment_dir', 
                        default=None,
                        help='folder to contain synthesized alignmeng')
    parser.add_argument('--use_gl', 
                        default=True,
                        type=str2bool,
                        help='Whether use GL to synthesize wav')
    parser.add_argument('--cmvn_path',
                        default='',
                        help='cmvn file')
    parser.add_argument('--minmax_path',
                        default=None,
                        help='full path to minmax.npz file')
    # Multi Speaker and multi style
    parser.add_argument('--speaker_id',
                        type=int,
                        default=-1)
    parser.add_argument('--style_id',
                        type=int,
                        default=-1)
    parser.add_argument('--emotion_id',
                        type=int,
                        default=-1)
    parser.add_argument('--speaker_map',
                        default=None)
    parser.add_argument('--style_map',
                        default=None)
    # Style Transfer/Control, GST/VAE
    parser.add_argument('--speaker_reference',
                        default=None,
                        help='speaker reference mel')
    parser.add_argument('--style_reference',
                        default=None,
                        help='style reference mel')
    parser.add_argument('--style_transfer',
                        default=False,
                        type=str2bool,
                        help='Whether use style transfer')
    parser.add_argument('--ref_acoustic_dir',
                        default=None,
                        help='Folder to contain reference acoustic spectrograms')
    parser.add_argument('--style_lab_dir',
                        default=None,
                        help='Folder to contain label for style control')
    # QuickSpeech
    parser.add_argument('--duration_alpha',
                        default=1.0,
                        type=float,
                        help='duration_alpha for QuickSpeech')
    # Pitch control
    parser.add_argument('--phone_pitch_dir',
                        default=None,
                        help='Path to phone-pitch file for pitch transfer')
    parser.add_argument('--phone_pitch',
                        default=None,
                        help='Specific phone-pitch file for pitch transfer')
    parser.add_argument('--pitch_alpha',
                        default=1.0,
                        type=float,
                        help='pitch_alpha for QuickPitch and TacotronPitch')
    # Denoise
    parser.add_argument('--ref_speaker_dir',
                        default=None,
                        help='Folder to contain reference speaker spectrograms')
    parser.add_argument('--ref_residual_dir',
                        default=None,
                        help='Folder to contain reference residual spectrograms')
    # SPSS
    parser.add_argument('--state_duration_dir',
                        default=None,
                        help='state_duration dir which contains predicted or ground-truth state duration')
    args = parser.parse_args()


    hparams = YParams(args.yaml_conf)
    modified_hp = hparams.parse(args.hparams)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.alignment_dir is not None:
        os.makedirs(args.alignment_dir, exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    synthesizer = Synthesizer(hparams, args)
    synthesizer()


if __name__ == '__main__':
    main()
