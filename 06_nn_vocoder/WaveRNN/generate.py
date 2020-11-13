import torch
import argparse
import soundfile as sf
import os
import time
import numpy as np
from models.model import Model


def attempt_to_restore(model, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint["model"])


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def create_model(args):
    model = Model(quantization_channels=args.quantization_channels,
                  gru_channels=896,
                  fc_channels=896,
                  lc_channels=args.local_condition_dim,
                  upsample_factor=(5, 5, 8),
                  use_gru_in_upsample=True).cuda()

    return model


def synthesis(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    model = create_model(args).to(device)
    if args.resume is not None:
        attempt_to_restore(model, args.resume, args.use_cuda)
    model.after_update()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, 'valid_scp'), 'r') as f:
        for line in f:
            mel_path = line.strip().replace(
                'labels', 'acoustic_features/mels').replace('lab', 'npy')
            name = os.path.basename(mel_path).replace('npy', 'wav')
            condition = np.load(mel_path)
            condition = torch.FloatTensor(condition).unsqueeze(0).to(device)
            print(f"Generating {name}")
            audio = model.generate(condition)
            sf.write('{}/{}'.format(output_dir, name),
                     audio.cpu().numpy().astype(np.int16), samplerate=16000)

    
def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/test',
                        help='Directory of tests data')
    parser.add_argument('--resume', type=str, default="ema_logdir")
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--quantization_channels', type=int, default=256)

    args = parser.parse_args()
    synthesis(args)


if __name__ == "__main__":
    main()
