import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq),
                                                                    max_words)])


def plot_alignment(alignments, text, _id, global_step, path):
    num_alignment = len(alignments)
    fig = plt.figure(figsize=(12, 16))
    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_alignment, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.suptitle(
        "record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}"
    )
    fig.savefig(path, format='png')
    plt.close()


def plot_spectrogram(pred_spectrogram, path, info=None, split_title=False,
                     target_spectrogram=None, max_len=None, auto_aspect=False):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if info is not None:
        if split_title:
            title = split_title_line(info)
        else:
            title = info

    fig = plt.figure(figsize=(10, 8))
    # Set common labels
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

    # target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram),
                            aspect='auto', interpolation='none')
        else:
            im = ax1.imshow(np.rot90(target_spectrogram), interpolation='none')

        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65,
                     orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram),
                        aspect='auto', interpolation='none')
    else:
        im = ax2.imshow(np.rot90(pred_spectrogram), interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def gray2rgb(inputs, n):
    data = tf.slice(inputs, [0, 0, 0, 0], [n, -1, -1, -1])
    colormap = np.asarray(
        [[0.0000, 0.0000, 0.5625], [0.0000, 0.0000, 0.6250],
         [0.0000, 0.0000, 0.6875], [0.0000, 0.0000, 0.7500],
         [0.0000, 0.0000, 0.8125], [0.0000, 0.0000, 0.8750],
         [0.0000, 0.0000, 0.9375], [0.0000, 0.0000, 1.0000],
         [0.0000, 0.0625, 1.0000], [0.0000, 0.1250, 1.0000],
         [0.0000, 0.1875, 1.0000], [0.0000, 0.2500, 1.0000],
         [0.0000, 0.3125, 1.0000], [0.0000, 0.3750, 1.0000],
         [0.0000, 0.4375, 1.0000], [0.0000, 0.5000, 1.0000],
         [0.0000, 0.5625, 1.0000], [0.0000, 0.6250, 1.0000],
         [0.0000, 0.6875, 1.0000], [0.0000, 0.7500, 1.0000],
         [0.0000, 0.8125, 1.0000], [0.0000, 0.8750, 1.0000],
         [0.0000, 0.9375, 1.0000], [0.0000, 1.0000, 1.0000],
         [0.0625, 1.0000, 0.9375], [0.1250, 1.0000, 0.8750],
         [0.1875, 1.0000, 0.8125], [0.2500, 1.0000, 0.7500],
         [0.3125, 1.0000, 0.6875], [0.3750, 1.0000, 0.6250],
         [0.4375, 1.0000, 0.5625], [0.5000, 1.0000, 0.5000],
         [0.5625, 1.0000, 0.4375], [0.6250, 1.0000, 0.3750],
         [0.6875, 1.0000, 0.3125], [0.7500, 1.0000, 0.2500],
         [0.8125, 1.0000, 0.1875], [0.8750, 1.0000, 0.1250],
         [0.9375, 1.0000, 0.0625], [1.0000, 1.0000, 0.0000],
         [1.0000, 0.9375, 0.0000], [1.0000, 0.8750, 0.0000],
         [1.0000, 0.8125, 0.0000], [1.0000, 0.7500, 0.0000],
         [1.0000, 0.6875, 0.0000], [1.0000, 0.6250, 0.0000],
         [1.0000, 0.5625, 0.0000], [1.0000, 0.5000, 0.0000],
         [1.0000, 0.4375, 0.0000], [1.0000, 0.3750, 0.0000],
         [1.0000, 0.3125, 0.0000], [1.0000, 0.2500, 0.0000],
         [1.0000, 0.1875, 0.0000], [1.0000, 0.1250, 0.0000],
         [1.0000, 0.0625, 0.0000], [1.0000, 0.0000, 0.0000],
         [0.9375, 0.0000, 0.0000], [0.8750, 0.0000, 0.0000],
         [0.8125, 0.0000, 0.0000], [0.7500, 0.0000, 0.0000],
         [0.6875, 0.0000, 0.0000], [0.6250, 0.0000, 0.0000],
         [0.5625, 0.0000, 0.0000], [0.5000, 0.0000, 0.0000]],
        dtype=np.float32)
    level = colormap.shape[0]
    data = ((data - tf.reduce_min(data)) /
            (tf.reduce_max(data) - tf.reduce_min(data)))
    idx = tf.reshape(tf.cast(data * (level-1), tf.int32), [-1])
    return tf.reshape(tf.gather(colormap, idx),
                      [tf.shape(data)[0], tf.shape(data)[1],
                       tf.shape(data)[2], 3])
