import os
import argparse
from datasets.preprocessor import *
from utils import audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        default='',
                        help='Hyperparameter overrides as a comma-separated '
                        'list of name=value pairs.')
    parser.add_argument('--yaml_conf',
                        default='yaml/hparams.yaml',
                        help='yaml files for configurations.')
    parser.add_argument('--label_dir', 
                        default=None,
                        help='Folder that contains labels')
    parser.add_argument('--label_type',
                        default="tacotron",
                        help='Type of labels, tacotron or spss')
    parser.add_argument('--wav_dir',
                        required=True,
                        help='Folder that contains wavs')
    parser.add_argument('--out_feature_dir',
                        required=True,
                        help='Folder that contains extracted acoustic features.')
    parser.add_argument('--valid_size',
                        type=int,
                        default=2,
                        help='number of validation samples')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=int(cpu_count()//2),
                        help='Number of Parallel jobs for processing data.')

    args = parser.parse_args()

    # Parse hyper-parameters from .yaml
    hparams = YParams(args.yaml_conf)

    # Parse custom parameters from args.hparams
    modified_hp = hparams.parse(args.hparams)

    # Prepare data
    preprocessor = Preprocessor(modified_hp, args)
    preprocessor()

if __name__ == '__main__':
    main()
