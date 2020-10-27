import tensorflow as tf
from ruamel.yaml import YAML as yaml
from tensorflow.contrib.training import HParams
import collections
import os
import argparse
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Unsupported value encountered.')

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

class YParams(HParams):

    def __init__(self, yaml_file):
        if not os.path.exists(yaml_file):
            raise IOError("yaml file: {} is not existed".format(yaml_file))
        super().__init__()
        self.d = collections.OrderedDict()
        with open(yaml_file) as fp:
            for _, v in yaml().load(fp).items():
                for k1, v1 in v.items():
                    try:
                        if self.get(k1):
                            self.set_hparam(k1, v1)
                        else:
                            self.add_hparam(k1, v1)
                        self.d[k1] = v1
                    except Exception:
                        import traceback
                        print(traceback.format_exc())

    # @property
    def get_elements(self):
        return self.d.items()

