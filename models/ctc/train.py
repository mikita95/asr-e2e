# coding=utf-8
import argparse
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

import nn.models as md

FLAGS = None

# Accounting the 0th index + space + blank label = 28 characters
num_classes = ord('я') - ord('а') + 1 + 1 + 1


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    from models.ctc import data_input
    feature_vectors, labels = data_input.inputs(data_dir=FLAGS.data_dir,
                                                max_examples_per_epoch=FLAGS.max_examples_per_epoch,
                                                batch_size=FLAGS.batch_size, feature_settings=vars(FLAGS))

    return feature_vectors, labels


def inference(feature_vectors, seq_lengths):
    return md.create_model(arch_type=FLAGS.model,
                           feature_input=feature_vectors,
                           seq_lengths=seq_lengths,
                           settings=None,
                           num_classes=num_classes,
                           mode=FLAGS.mode)


def compute_loss(logits, labels, seq_lengths):
    loss = ctc.ctc_loss(labels, logits, seq_lengths)
    cost = tf.reduce_mean(loss)
    tf.add_to_collection('losses', cost)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = FLAGS.max_examples_per_epoch / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * 350)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(0.1,
                                    global_step,
                                    decay_steps,
                                    0.1,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def run(_):
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get feature_vectors and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            feature_vectors, labels = inputs()

        seq_lengths = [feature_vectors.shape[1]] * feature_vectors.shape[0]
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = inference(feature_vectors, seq_lengths)

        # Calculate loss.
        loss = compute_loss(logits, labels, seq_lengths)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and test nn')

    parser.add_argument('--mode',
                        type=str,
                        help='Running mode',
                        choices=['train', 'test'],
                        default='train')

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to data dir')

    parser.add_argument('--val_path',
                        type=str,
                        help='Path to validation data dir')

    parser.add_argument('--model',
                        help='Name of neural network model',
                        type=str)

    parser.add_argument('--how_many_training_steps',
                        type=str,
                        default='15000,3000',
                        help='How many training loops to run')

    parser.add_argument('--eval_step_interval',
                        type=int,
                        default=400,
                        help='How often to evaluate the training results.')

    parser.add_argument('--learning_rate',
                        type=str,
                        default='0.001,0.0001',
                        help='How large a learning rate to use when training.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='How many items to train with at once')

    parser.add_argument('--max_examples_per_epoch',
                        type=int,
                        default=1000,
                        help='How many examples per epoch to train with')

    parser.add_argument('--summaries_dir',
                        type=str,
                        default='/tmp/retrain_logs',
                        help='Where to save summary logs for TensorBoard.')

    parser.add_argument('--train_dir',
                        type=str,
                        default='/tmp/speech_commands_train',
                        help='Directory to write event logs and checkpoint.')

    parser.add_argument('--save_step_interval',
                        type=int,
                        default=100,
                        help='Save model checkpoint every save_steps.')

    parser.add_argument('--start_checkpoint',
                        type=str,
                        default='',
                        help='If specified, restore this pretrained model before any training.')

    parser.add_argument('--check_nans',
                        type=bool,
                        default=False,
                        help='Whether to check for invalid numbers during processing')

    parser.add_argument("--mode",
                        help="Mode",
                        choices=['mfcc', 'fbank', 'raw'],
                        type=str,
                        default='mfcc')

    parser.add_argument("--format",
                        help="Format of files",
                        choices=['wav', 'mp3'],
                        type=str,
                        default='wav')

    parser.add_argument("--rate",
                        help="Sample rate of the audio files",
                        type=int,
                        default=16000)

    parser.add_argument("--channels",
                        help="Number of channels of the audio files",
                        type=int,
                        default=1)

    parser.add_argument("--winlen", type=float, default=0.025)
    parser.add_argument("--winstep", type=float, default=0.01)
    parser.add_argument("--numcep", type=int, default=13)
    parser.add_argument("--nfilt", type=int, default=26)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--lowfreq", type=int, default=0)
    parser.add_argument("--highfreq", type=int, default=None)
    parser.add_argument("--ceplifter", type=int, default=22)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--appendEnergy", type=bool, default=True)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
