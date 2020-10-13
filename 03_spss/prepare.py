
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import tensorflow as tf
from utils import *

def calculate_cmvn( label_dict, acoustic_dict, data_dim=None, out_file=''):
    """Calculate mean and var."""
    
    tf.logging.info("Calculating mean and var ")

    inputs_frame_count, targets_frame_count = 0, 0
    for key in tqdm(acoustic_dict):
        if data_dim is None:
            targets = np.loadtxt(acoustic_dict[key]).astype(np.float32)
        else:
            targets, _ = read_binary_file(acoustic_dict[key], data_dim)
        inputs = np.load(label_dict[key])
        if inputs_frame_count == 0:  # create numpy array for accumulating
            ex_inputs = np.sum(inputs, axis=0)
            ex2_inputs = np.sum(inputs ** 2, axis=0)
            ex_targets = np.sum(targets, axis=0)
            ex2_targets = np.sum(targets ** 2, axis=0)
        else:
            ex_inputs += np.sum(inputs, axis=0)
            ex2_inputs += np.sum(inputs ** 2, axis=0)
            ex_targets += np.sum(targets, axis=0)
            ex2_targets += np.sum(targets ** 2, axis=0)
        inputs_frame_count += len(inputs)
        targets_frame_count += len(targets)

    mean_inputs = ex_inputs / inputs_frame_count
    stddev_inputs = np.sqrt(ex2_inputs / inputs_frame_count - mean_inputs ** 2)
    stddev_inputs[stddev_inputs < 1e-20] = 1e-20

    mean_targets = ex_targets / targets_frame_count
    stddev_targets = np.sqrt(ex2_targets / targets_frame_count - mean_targets ** 2)
    stddev_targets[stddev_targets < 1e-20] = 1e-20

    np.savez(out_file,
                mean_inputs=mean_inputs,
                stddev_inputs=stddev_inputs,
                mean_targets=mean_targets,
                stddev_targets=stddev_targets)


def main():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    os.makedirs("cmvn", exist_ok=True)

    acoustic_label_dir = "train_data/acoustic_features"
    acoustic_target_dir = "train_data/acoustic_targets"
    duration_label_dir = "train_data/duration_features"
    duration_target_dir = "train_data/duration_targets"

    with open("train.scp", 'r') as file_list:
        file_id = file_list.readlines()
        for i in range(len(file_id)):
            file_id[i] = file_id[i].strip()

    # file_id=file_id[:1000]

    print("load acoustic data ! ")
    label_dict = {}
    acoustic_dict = {}
    for key in tqdm(file_id):
        # print(os.path.join(acoustic_label_dir, key + ".npy"))
        label_dict[key] = os.path.join(acoustic_label_dir, key + ".npy")
        acoustic_dict[key] = os.path.join(acoustic_target_dir, key + ".cmp")

    print("prepare acoustic data ! ")
    calculate_cmvn(label_dict=label_dict, acoustic_dict=acoustic_dict, data_dim=75, out_file="cmvn/train_cmvn_spss")


    print("load duration data ! ")
    label_dict = {}
    acoustic_dict = {}
    for key in tqdm(file_id):
        label_dict[key] = os.path.join(duration_label_dir, key + ".npy")
        acoustic_dict[key] = os.path.join(duration_target_dir, key + ".lab")

    print("prepare duration data ! ")
    calculate_cmvn(label_dict=label_dict, acoustic_dict=acoustic_dict, out_file="cmvn/train_cmvn_dur")


if __name__ == '__main__':
    main()

