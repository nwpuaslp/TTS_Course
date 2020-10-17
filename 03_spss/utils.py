# coding=utf-8
""" """
import numpy as np
import tensorflow as tf
from hparam import  hparams

def read_binary_file(filename, dimension=None):
    """Read data from matlab binary file (row, col and matrix).

    Returns:
        A numpy matrix containing data of the given binary file.
    """
    fid_lab = open(filename, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data' % (
        dimension)
    features = features[:(dimension * (features.size // dimension))]
    features = features.reshape((-1, dimension))
    return features, features.shape[0] 

def write_binary_file(data, output_file_name):
        data = np.asarray(data, np.float32)
        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

def pending_state_info(onehot_label, state_duration):
    nPhoneme = onehot_label.shape[0]
    label_dim = onehot_label.shape[1]
    assert (onehot_label.shape[0] == state_duration.shape[0])
    nState = state_duration.shape[1]
    offset = 0
    total_frames = 0
    phone_duration = np.zeros([nPhoneme])
    for p in range(nPhoneme):
        for state in range(nState):
            if state_duration[p, state] < 0:
                state_duration[p, state] = 0
            phone_duration[p] += int(state_duration[p, state] + 0.5)
    total_frames = np.sum(phone_duration)

    linguistic_feats = add_state_info(
        onehot_label, state_duration, phone_duration, label_dim).astype(np.float32)
    return linguistic_feats

def add_state_info(oneHot, state_duration, phone_duration, label_dim):
    total_frames = np.sum(phone_duration).astype(np.int)
    linguistic_feats = np.zeros([total_frames, label_dim + 9])

    nState = state_duration.shape[1]
    nPhoneme = state_duration.shape[0]
    flag = 0
    # 填充前面288维度
    for i in range(nPhoneme):
        dur = phone_duration[i].astype(np.int)
        for tmp in range(dur):
            linguistic_feats[flag, :label_dim] += oneHot[i]
            flag += 1
    assert flag == total_frames
    flag = 0
    # 填充后面9维度
    for i in range(nPhoneme):
        state_duration_base = 0
        for s in range(nState):
            cur_phone_duration = phone_duration[i]
            state_frames = int(state_duration[i, s] + 0.5)
            for f in range(state_frames):
                oneHot_dim = oneHot.shape[1]
                # 1- (f+1) 当前帧是状态的第几帧， state_frames, 当前状态总共有几帧
                linguistic_feats[flag, oneHot_dim] = float(f + 1) / float(state_frames)
                oneHot_dim += 1
                # 2- 当前帧是状态帧的倒数第几帧
                linguistic_feats[flag, oneHot_dim] = float(state_frames - f) / float(state_frames)
                oneHot_dim += 1
                #  3- 当前帧所在的状态总共有几帧
                linguistic_feats[flag, oneHot_dim] = float(state_frames)
                oneHot_dim += 1
                # 4- 当前帧所属的状态
                linguistic_feats[flag, oneHot_dim] = float(s + 1)
                oneHot_dim += 1
                # 5- 当前帧属于倒数第几个状态
                linguistic_feats[flag, oneHot_dim] = float(nState - s)
                oneHot_dim += 1
                # 6- 当前帧所处的音素的总时长
                linguistic_feats[flag, oneHot_dim] = cur_phone_duration
                oneHot_dim += 1
                # 7- 音素当前状态的帧数与音素的总帧数的比例
                linguistic_feats[flag, oneHot_dim] = float(state_frames) / cur_phone_duration
                oneHot_dim += 1
                linguistic_feats[flag, oneHot_dim] = float(
                    cur_phone_duration - f - state_duration_base) / cur_phone_duration
                oneHot_dim += 1
                linguistic_feats[flag, oneHot_dim] = float(state_duration_base + f + 1) / cur_phone_duration
                oneHot_dim += 1
    return linguistic_feats

