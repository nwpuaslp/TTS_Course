from datasets.label_loader import *
from datasets.acoustic_extractor import *
from  datasets.preprocessor import *
import os
import numpy as np
import shutil

class Preprocessor():
    """
    Label loader for all types of labels.

    When you want to load another label format, there two step you need to do:
        1) Add your label type (e.g CustomLabel) into 'supported_label_type' in hparams.yaml
        2) Add it into LabelLoader.supported_load_functions and write your own loading function.

    Noted that you can also add other informations like speaker id or style id in this step.
    """
    def __init__(self, hparams, args):
        super(Preprocessor, self).__init__()
        self.hparams = hparams
        self.args = args
        self.preprocess_type = hparams.preprocess_type
        assert (self.preprocess_type in hparams.supported_preprocess_type)
        self.supported_proc_functions = {"BasicProcessor": self.BasePreprocessor,
                                         "AcousticProcessor": self.AcousticProcessor,
                                         "CustomProcessor": self.CustomProcessor}
        self.process_function = self.supported_proc_functions[self.preprocess_type]

    def __call__(self):
        self.process_function()

    def BasePreprocessor(self):

        label_loader = LabelLoader(self.hparams, self.args)
        label_dict = label_loader()

        # Extract acoustic features
        acoustic_extractor = FeatureExtractor(self.hparams, self.args)
        acoustic_dict = acoustic_extractor(label_dict)

       
    def AcousticProcessor(self):
        # Get wav keys
        wav_keys = [os.path.splitext(os.path.basename(filename))[0] 
            for filename in os.listdir(self.args.wav_dir)]
        # Extract acoustic features
        acoustic_extractor = FeatureExtractor(self.hparams, self.args)
        acoustic_dict = acoustic_extractor(wav_keys)

    def CustomProcessor(self):
        raise NotImplementedError(
            "You need to implement the details of your own data loader")

