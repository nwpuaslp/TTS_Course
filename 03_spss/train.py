# coding=utf-8
""" """
import argparse
import os
import traceback
from datetime import datetime
import time

import numpy as np

from model import *
import tensorflow as tf

from hparam import hparams
from  data_reader import *
_format = '%Y-%m-%d %H:%M:%S.%f'

def main_function(hparams, output_dir, training_records,
                  valid_records, restore_path="", checkpoint=None):
    tensorboard_dir = os.path.join(output_dir, 'events')
    model_dir = os.path.join(output_dir, "model")
    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    tf.set_random_seed(hparams.random_seed)
    coord = tf.train.Coordinator()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if(args.model_type == 'AcousticModel'):
        placeholders = [
            tf.placeholder(tf.float32, [None, None, hparams.acoustic_label_dim], 'labels'),
            tf.placeholder(tf.int32, [None], 'input_length'),
            tf.placeholder(tf.float32, [None, None, hparams.acoustic_dim], 'targets')
        ]
    elif(args.model_type == 'DurationModel'):
        placeholders = [
            tf.placeholder(tf.float32, [None, None, hparams.dur_label_dim], 'labels'),
            tf.placeholder(tf.int32, [None], 'input_length'),
            tf.placeholder(tf.float32, [None, None, hparams.dur_dim], 'targets')
        ]
    else:
        raise 'model type error ! '

    with tf.device('/cpu:0'):
        with tf.name_scope('inputs'):
            if(args.model_type == 'AcousticModel'):
                reader = DataReader_Acoustic(coord, args.filelist, acoustic_features_dir=args.acoustic_features_dir, labels_dir=args.labels_dir, hparams=hparams)
            elif(args.model_type == 'DurationModel'):
                reader = DataReader_Duration(coord, args.filelist, acoustic_features_dir=args.acoustic_features_dir, labels_dir=args.labels_dir, hparams=hparams)
            else:
                raise 'model_type error ! '

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        if(args.model_type == 'AcousticModel'):
            model = AcousticModel(hparams, placeholders, is_training=True)
        elif(args.model_type == 'DurationModel'):
            model = DurationModel(hparams, placeholders, is_training=True)

        model.add_loss(global_step)
        model.add_optimizer(global_step)
        training_statistics = model.add_tensorboard_stats()

    step = 0
    global_saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.keep_checkpoint_max)
    restore_saver = tf.train.Saver(tf.trainable_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(
                tensorboard_dir, sess.graph)
            reader.start_threads()
            sess.run(init_ops)

            # Restore model
            if restore_path != "":
                print("Resuming trained model from {}.".format(restore_path))
                restore_saver.restore(sess, restore_path)

            while (not coord.should_stop() and step <
                   hparams.total_training_steps):

                start_time = time.time()
                features = reader.dequeue_tts(num_elements=hparams.batch_size)
                dicts = dict()
                for i in range(len(placeholders)):
                    dicts[placeholders[i]] = features[i]
                step, loss, opt = sess.run(
                    [global_step, model.loss, model.optimize],feed_dict=dicts)

                message = "{:s}: Step {:7d}  loss={:.5f}".format(
                    (datetime.now().strftime(_format)[:-3]), step, loss)
                print(message)

                if loss > 1000 or np.isnan(loss):
                    print('Loss exploded to {:.5f} at step {}'.format(
                        loss, step))
                    raise Exception('Loss exploded')
                # Summary writer
                if step % hparams.save_summary_steps == 0 :
                    print('\nWriting training summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(training_statistics, feed_dict=dicts), step)
                    summary_writer.flush()


                # Statistics on training set
                if (step % hparams.save_checkpoints_steps == 0 or step ==
                    hparams.total_training_steps):
                    global_saver.save(sess, checkpoint_path, global_step=global_step)


        except Exception as e:
            print('Exiting due to exception: {}'.format(e))
            traceback.print_exc()
            coord.request_stop(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        default="logdir",
                        help='Directory where to save logs and results.')
    parser.add_argument('--checkpoint',
                        default="",
                        help='Checkpoint path for restoring.')
    parser.add_argument('--filelist',
                        required=True)
    parser.add_argument('--acoustic_features_dir',
                        required=True)
    parser.add_argument('--labels_dir',
                        required=True)
    parser.add_argument('--model_type',
                        default="AcousticModel")

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    main_function(hparams,
                  args.log_dir,
                  None,
                  None,
                  restore_path=args.checkpoint,
                  checkpoint=args.checkpoint)
