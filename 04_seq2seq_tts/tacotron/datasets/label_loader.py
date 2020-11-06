import os
import numpy as np
from tqdm import tqdm
from text.phones_mix import phone_to_id

class LabelLoader():
    """
    Label loader for labels.
    Noted that you can also add other informations like speaker id or style id in this step.
    """
    def __init__(self, hparams, args):
        super(LabelLoader, self).__init__()
        self.hparams = hparams
        self.args = args
        self.label_dir = args.label_dir

    def load_label(self, label_path):
        # Load a single tacotron label
        with open(label_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            meta_list = []
            content = line.strip().split('|')[2].split(' ')
            for i,item in enumerate(content):
                meta_list.append(item)

            if len(meta_list) < 1:
                raise ValueError("The content of file {} is empty.".format(label_path))
            return meta_list

    def __call__(self):
        label_dict = {}
        for filename in tqdm(sorted(os.listdir(self.label_dir))):
            if not filename.endswith('.lab'):
                raise ValueError("The {} may not be a label file.".format(filename))
            label_path = os.path.join(self.label_dir, filename.strip())
            label_dict[filename.split('.')[0]] = self.load_label(label_path)
        # Check related wavs
        for key in list(label_dict.keys()):
            wav_path = os.path.join(self.args.wav_dir, key + '.wav')
            if not os.path.exists(wav_path):
                print("Wave file {} doesn't exists, ignore the related label.".format(wav_path))
                label_dict.pop(key)
        return label_dict

