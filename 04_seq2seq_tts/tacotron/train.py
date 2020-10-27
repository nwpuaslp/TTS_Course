import time
import traceback
from datetime import datetime

from datasets.data_reader import *
from models.tacotron import *

log = infolog.log
_format = '%Y-%m-%d %H:%M:%S.%f'


def get_graph_stats(graph):
    flops = tf.profiler.profile(graph,
                                options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph,
                                 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    return flops, params


def train(hparams, output_dir, restore_path=""):
    tensorboard_dir = os.path.join(output_dir, 'events')
    model_dir = os.path.join(output_dir, "Tacotron")
    checkpoint_path = os.path.join(model_dir, "Tacotron.ckpt")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    tf.set_random_seed(hparams.random_seed)
    coord = tf.train.Coordinator()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_placeholders = [
        tf.placeholder(tf.int32, [None, None], 'phones'),
        tf.placeholder(tf.int32, [None], 'input_length'),
        tf.placeholder(tf.float32, [None, None, hparams.acoustic_dim],
                       'acoustic_targets'),
        tf.placeholder(tf.float32, [None, None], 'stop_token_targets'),
        tf.placeholder(tf.int32, [None], 'targets_length'),
    ]
    with tf.device('/cpu:0'):
        with tf.name_scope('inputs'):
            trainreader = DataReader(coord, args.train_filelist, acoustic_features_dir=args.acoustic_features_dir,
                                labels_dir=args.labels_dir, hparams=hparams, args=args)


    # Build model
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        model = TacotronModel(train_placeholders, hparams, is_training=True)
        model.init_model(global_step)
        model.add_loss(global_step)
        model.add_optimizer(global_step)
        training_statistics = model.add_training_stats()

    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    global_saver = tf.train.Saver(tf.global_variables(), max_to_keep=hparams.keep_checkpoint_max)

    log("Model training set to a maximum of {} steps".format(
        hparams.total_training_steps))

    # Config TensorFlow GPU options
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
    # Train and evaluate
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(
                tensorboard_dir, sess.graph)
            trainreader.start_threads()

            sess.run(init_ops)

            flops, params = get_graph_stats(sess.graph)
            log("FLOPs: {}".format(flops.total_float_ops))
            log("Trainable params: {}".format(params.total_parameters))

            while (not coord.should_stop() and step <
                   hparams.total_training_steps):

                start_time = time.time()
                train_features = trainreader.dequeue_tts(num_elements=hparams.batch_size)
                
                train_dicts = dict()

                for i in range(len(train_placeholders)):
                    train_dicts[train_placeholders[i]] = train_features[i]


                step, loss, opt = sess.run(
                    [global_step, model.loss, model.optimize], feed_dict = train_dicts)
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = "{:s}: Step {:7d} [{:.3f} s/step, loss={:.5f}, avg_loss={:.5f}]".format(
                    (datetime.now().strftime(_format)[:-3]), step,
                    time_window.average, loss, loss_window.average)
                log(message, end='\r', slack=(
                        step % hparams.save_checkpoints_steps == 0))

                if loss > 1000 or np.isnan(loss):
                    log('Loss exploded to {:.5f} at step {}'.format(
                        loss, step))
                    raise Exception('Loss exploded')
                # Summary writer
                if step % hparams.save_summary_steps == 0:
                    log('\nWriting training summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(training_statistics, feed_dict=train_dicts), step)
                    summary_writer.flush()

                # Statistics on training set
                if (step % hparams.save_checkpoints_steps == 0 or step ==
                    hparams.total_training_steps):
                    global_saver.save(sess, checkpoint_path, global_step=global_step)

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_conf',
                        default='hparams.yaml',
                        help='yaml files for configurations.')
    parser.add_argument('--hparams', default='',
                        help='Overrides hyper parameters as a comma-separated '
                             'list of name=value pairs.')
    parser.add_argument('--log_dir',
                        default="",
                        help='Directory where to save logs and results.')
    parser.add_argument('--train_filelist',
                        required=True)
    parser.add_argument('--valid_filelist',
                        required=True)
    parser.add_argument('--acoustic_features_dir',
                        required=True)
    parser.add_argument('--labels_dir',
                        required=True)
    args = parser.parse_args()

    # Parse hyper-parameters from .yaml
    hparams = YParams(args.yaml_conf)
    modified_hp = hparams.parse(args.hparams)

    if args.log_dir == "":
        args.log_dir = "log"
    os.makedirs(args.log_dir, exist_ok=True)

    # Record hyperparameters
    infolog.init(
        os.path.join(args.log_dir, 'train.log'),
        "Tacotron", None)

    train(modified_hp, args.log_dir)
