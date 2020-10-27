import numpy as np
import tensorflow as tf

def item_or_tuple(seq,is_multi):
    t = tuple(seq)
    if is_multi:
        return t
    else:
        return t[0]

def _groupby(input_labels):
    """Squeeze the input label sequence
    Args:
      input_labels: [B, T]
    Return:
      squeezed_labels: [B, T_reduced]
      retrieve_key: [B, T]
    """
  
    batch_size = input_labels.shape[0]
    T = input_labels.shape[1]
  
    L = []
    R = []
    max_reduced_len = 0
    lenghs = []
    D = []
    for batch in range(batch_size):
        label = input_labels[batch]
        label_uniq = []
        rkey = []
        key_t = 0
        dur_list = []
        for t in range(T):
            if t == 0:
                label_uniq.append(label[0])
                rkey.append(key_t)
                dur = 1
                continue

            if (label[t] == label_uniq[-1]):
                rkey.append(key_t)
                dur += 1
            else:
                label_uniq.append(label[t])
                key_t = key_t + 1
                rkey.append(key_t)
                dur_list.append(dur)
                dur = 1
        dur_list.append(dur)

        L.append(label_uniq)
        R.append(rkey)
        D.append(dur_list)
        lenghs.append(len(label_uniq))
        if len(label_uniq) > max_reduced_len:
            max_reduced_len = len(label_uniq)

    for batch in range(batch_size):
        label_uniq_arr = np.array(L[batch])
        dur_arr = np.array(D[batch])
        padnum = max_reduced_len - len(L[batch])
        L[batch] = np.pad(label_uniq_arr, (0, padnum), mode='constant', constant_values=0)
        R[batch] = [batch*max_reduced_len + v for v in R[batch]]
        D[batch] = np.pad(dur_arr, (0, padnum), mode='constant', constant_values=0)
  
    L = np.array(L, dtype=np.int32)
    R = np.array(R, dtype=np.int32)
    Len = np.array(lenghs, dtype=np.int32)
    D = np.array(D, dtype=np.int32)
  
    return L, R, Len, D

def groupby(input_labels, batch_size):
    """Squeeze the input label sequence
    E.g., input_labels is [1,1,1,2,2,2], we want to retrun
          label: [1,2], also the location of each item in input_labels
          in new labels e.g., [0,0,0,1,1,1]
    Args:
      input_labels: [N, T]
    Return:
      labels: [N, T_reduced]
      map_key: [N, T]
    """
    labels, map_key, lengths, durations = tf.py_func(_groupby, [input_labels], [tf.int32, tf.int32, tf.int32, tf.int32])
    labels.set_shape([batch_size, None])
    durations.set_shape([batch_size, None])
    map_key.set_shape(input_labels.get_shape())
    lengths.set_shape([batch_size])
  
    return labels, map_key, lengths, durations


def pad_ali(ali_keys, r):

    def _pad_ali(x):
        max_len = x.shape[1]
        remainder = max_len % r
        if remainder == 0:
            pad_num = 0
        else:
            pad_num = r - remainder
        _pad = x[0,-1]
        y = np.pad(x, ((0,0),(0, int(pad_num))), mode='constant', constant_values=_pad)
        y.astype(int)
        return y
  
    ali_keys = tf.py_func(_pad_ali, [ali_keys], tf.int32)
    return ali_keys

def get_tensor_shape(x):
   """Return list of dims, statically where possible."""
   x = tf.convert_to_tensor(x)

   # If unknown rank, return dynamic shape
   if x.get_shape().dims is None:
       return tf.shape(x)

   static = x.get_shape().as_list()
   shape = tf.shape(x)

   ret = []
   for i in range(len(static)):
       dim = static[i]
       if dim is None:
           dim = shape[i]
       ret.append(dim)
   return ret

def round_up(x, multiple):
    """TF version of remainder = x % multiple"""
    remainder = tf.mod(x, multiple)
    # Tf version of return x if remainder == 0 else x + multiple - remainder
    x_round = tf.cond(
        tf.equal(remainder, tf.zeros(tf.shape(remainder), dtype=tf.int32)),
        lambda: x, lambda: x + multiple - remainder)

    return x_round

def compute_mask(lengths, r, expand_dim=True):
    max_len = tf.reduce_max(lengths)
    max_len = round_up(max_len, tf.convert_to_tensor(r))
    if expand_dim:
        return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len,
        dtype=tf.float32), axis=-1)
    return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def mask_feature(x, lengths=None, expand_dim=False, mask=None):
    if  (lengths is None) + (mask is None) != 1:
        raise ValueError("Lengths and mask cannot be all None or with value during masking.")
    if lengths is not None:
            mask = compute_mask(lengths, 1, expand_dim=True)
    if expand_dim:
        mask = tf.expand_dims(mask, axis=-1)
    return x * mask

def MaskedMSE(targets, outputs, targets_lengths, hparams, mask=None):
    '''Computes a masked Mean Squared Error
    '''
    if mask is None:
        mask = compute_mask(targets_lengths, hparams.outputs_per_step, True)

    #[batch_size, time_dimension, channel_dimension(mels)]
    ones = tf.ones(shape=[tf.shape(mask)[0], tf.shape(mask)[
                   1], tf.shape(targets)[-1]], dtype=tf.float32)
    mask_ = mask * ones

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets),
     tf.shape(mask_))]):
        return tf.losses.mean_squared_error(labels=targets, predictions=outputs,
         weights=mask_)


def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams,
 mask=None):
    '''Computes a masked SigmoidCrossEntropy with logits
    '''
    if mask is None:
        mask = compute_mask(targets_lengths, hparams.outputs_per_step, False)

    with tf.control_dependencies([tf.assert_equal(tf.shape(targets),
    tf.shape(mask))]):
        losses = tf.nn.weighted_cross_entropy_with_logits(
            targets=targets, logits=outputs,
            pos_weight=hparams.cross_entropy_pos_weight)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask),
     tf.shape(losses))]):
        masked_loss = losses * mask

    return (tf.reduce_sum(masked_loss) / tf.count_nonzero(mask,
        dtype=tf.float32))
