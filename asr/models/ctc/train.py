# coding=utf-8
import argparse
import time
from datetime import datetime

import tensorflow as tf

import nn as mb
from models import Mode
from utils import writer

FLAGS = None


def get_loss(feats, labels, seq_lens):
    logits = mb.create_model(arch_type=FLAGS.model,
                             feature_input=feats,
                             seq_lengths=seq_lens,
                             batch_size=FLAGS.batch_size,
                             mode=Mode.TRAIN,
                             num_classes=len(writer.ALPHABET) + 1,
                             config_file=FLAGS.config_file)

    ctc_loss = tf.nn.ctc_loss(inputs=logits,
                              labels=labels,
                              sequence_length=seq_lens)

    ctc_loss_mean = tf.reduce_mean(ctc_loss)

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_lens)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

    tf.summary.scalar('ctc_loss_mean', ctc_loss_mean)
    tf.summary.scalar('ler', ler)
    
    return ctc_loss_mean, ler


def set_learning_rate():
    """ Set up learning rate schedule """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed
    global_step = tf.get_variable(name='global_step',
                                  shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    decay_steps = int(FLAGS.num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.initial_lr,
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=FLAGS.lr_decay_factor,
                                               staircase=True)

    return learning_rate, global_step


def fetch_data():
    """ Fetch features, labels and sequence_lengths from a common queue."""

    feats, labels, seq_lens = src.models.ctc.input.inputs(tfrecords_path=FLAGS.record_path,
                                                          batch_size=FLAGS.batch_size,
                                                          shuffle=FLAGS.shuffle)

    return feats, labels, seq_lens


def run_train_loop(sess, operations, saver):
    import numpy as np
    import os
    """ Train the model for required number of steps."""
    (train_op, loss_op, summary_op, ler_op) = operations
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # Evaluate the ops for max_steps
    for step in range(FLAGS.max_steps):
        start_time = time.time()

        loss_value, _, ler_value = sess.run([loss_op, train_op, ler_op])

        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Print progress periodically.
        if step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / duration
            format_str = ('%s: step %d, '
                          'ler = %.2f, '
                          'loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, ler_value, loss_value, examples_per_sec, duration))

        # Run the summary ops periodically.
        if step % 50 == 0:
            summary_writer.add_summary(sess.run(summary_op), step)

        # Save the model checkpoint periodically.
        if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""
    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        checkpoint_path = ckpt.model_checkpoint_path
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        print('No checkpoint file found')
        return


def add_summaries(summaries, learning_rate, train_op):
    """ Add summary ops"""

    # Track quantities for Tensorboard display
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))

    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)
    return summary_op


def train():
    """Train deepSpeech for a number of steps.
    This function build a set of ops required to build the model and optimize
    weights.
    """
    with tf.Graph().as_default():

        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Fetch a batch worth of data for each tower
        data = fetch_data()

        [feats, labels, seq_lens] = data

        loss_op, ler_op = get_loss(feats, labels, seq_lens)
        train_op = optimizer.minimize(loss_op)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # Build summary op
        summary_op = add_summaries(summaries, learning_rate, train_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        # Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the
        # ops do not have GPU implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        # Initialize vars.
        if FLAGS.checkpoint is not None:
            global_step = initialize_from_checkpoint(sess, saver)
        else:
            sess.run(tf.initialize_all_variables())

        # Start the queue runners.
        tf.train.start_queue_runners(sess)

        # Run training loop
        run_train_loop(sess, (train_op, loss_op, summary_op, ler_op), saver)


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and tests nn')

    num_gpus = len([x for x in device_lib.list_local_devices()
                    if x.device_type == "GPU"])

    parser.add_argument('--record_path',
                        type=str,
                        help='Path to data dir')

    parser.add_argument('--val_path',
                        type=str,
                        help='Path to validation data dir')

    parser.add_argument('--model',
                        help='Name of neural network model',
                        type=str)

    parser.add_argument('--config_file',
                        help='Configs of neural network model',
                        default=None,
                        type=str)

    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
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

    parser.add_argument('--num_batches_per_epoch',
                        type=int,
                        default=100)

    parser.add_argument('--max_steps',
                        type=int,
                        default=20000,
                        help='Number of batches to run')

    parser.add_argument('--log_device_placement',
                        type=bool,
                        default=False,
                        help='Whether to log device placement')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle',
                                dest='shuffle',
                                action='store_true')
    feature_parser.add_argument('--no-shuffle',
                                dest='shuffle',
                                action='store_false')
    parser.set_defaults(shuffle=True)

    parser.add_argument('--keep_prob',
                        type=float,
                        default=0.5,
                        help='Keep probability for dropout')

    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Continue training from checkpoint file')

    parser.add_argument('--initial_lr',
                        type=float,
                        default=0.00001,
                        help='Initial learning rate for training')

    parser.add_argument('--moving_avg_decay',
                        type=float,
                        default=0.9999,
                        help='Decay to use for the moving average of weights')

    parser.add_argument('--num_epochs_per_decay',
                        type=int,
                        default=5,
                        help='Epochs after which learning rate decays')

    parser.add_argument('--lr_decay_factor',
                        type=float,
                        default=0.9,
                        help='Learning rate decay factor')

    FLAGS, unparsed = parser.parse_known_args()
    train()
