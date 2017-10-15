# coding=utf-8
import argparse
import time
from datetime import datetime

import tensorflow as tf

import asr.nn.model as mb
from asr.models.params.modes import Mode

FLAGS = None


def get_loss(feats, labels, seq_lens):
    import asr.models.ctc.labels.handler as hn

    logits = mb.create_model(arch_type=FLAGS['model'],
                             feature_input=feats,
                             seq_lengths=seq_lens,
                             batch_size=int(FLAGS['batch_size']),
                             mode=Mode.TRAIN,
                             num_classes=hn.CTCLabelsHandler(
                                 alphabet_file=FLAGS['alphabet_config_file']).get_alphabet_size() + 1,
                             config_file=FLAGS['model_config_file'])

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
    global_step = tf.get_variable('global_step',
                                  [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Calculate the learning rate schedule.
    decay_steps = int(FLAGS['num_batches_per_epoch']) * int(FLAGS['num_epochs_per_decay'])

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(learning_rate=float(FLAGS['initial_lr']),
                                               global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=float(FLAGS['lr_decay_factor']),
                                               staircase=True)

    return learning_rate, global_step


def fetch_data():
    import asr.models.ctc.input
    """ Fetch features, labels and sequence_lengths from a common queue."""

    feats, labels, seq_lens = asr.models.ctc.input.inputs(tfrecords_path=FLAGS['train_record_path'],
                                                          batch_size=int(FLAGS['batch_size']),
                                                          shuffle=bool(FLAGS['shuffle']))

    return feats, labels, seq_lens


def run_train_loop(sess, operations, saver):
    import numpy as np
    import os
    """ Train the model for required number of steps."""
    (train_op, loss_op, summary_op, ler_op) = operations
    summary_writer = tf.summary.FileWriter(FLAGS['train_dir'], sess.graph)

    # Evaluate the ops for max_steps
    for step in range(int(FLAGS['max_steps'])):
        start_time = time.time()

        loss_value, _, ler_value = sess.run([loss_op, train_op, ler_op])

        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Print progress periodically.
        if step % 10 == 0:
            examples_per_sec = int(FLAGS['batch_size']) / duration
            format_str = ('%s: step %d, '
                          'ler = %.2f, '
                          'loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, ler_value, loss_value, examples_per_sec, duration))

        # Run the summary ops periodically.
        if step % 50 == 0:
            summary_writer.add_summary(sess.run(summary_op), step)

        # Save the model checkpoint periodically.
        if step % 100 == 0 or (step + 1) == int(FLAGS['max_steps']):
            checkpoint_path = os.path.join(FLAGS['train_dir'], 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""
    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(FLAGS['checkpoint'])
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
    import asr.models.params.optimizers as opt
    with tf.Graph().as_default(), tf.device('/cpu'):

        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = opt.get(FLAGS['optimizer'])(learning_rate)

        # Fetch a batch worth of data for each tower
        data = fetch_data()

        [feats, labels, seq_lens] = data

        loss_op, ler_op = get_loss(feats, labels, seq_lens)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

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
            log_device_placement=bool(FLAGS['log_device_placement'])))

        # Initialize vars.

        sess.run(tf.initialize_all_variables())

        # Start the queue runners.
        tf.train.start_queue_runners(sess)

        # Run training loop
        run_train_loop(sess, (train_op, loss_op, summary_op, ler_op), saver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and tests nn')

    parser.add_argument('--train_config',
                        type=str,
                        help='Path to train config file')

    ARGS, unparsed = parser.parse_known_args()
    import configparser

    config = configparser.ConfigParser()
    config.read(ARGS.train_config, encoding='utf8')
    FLAGS = dict(config.items('CONFIGS'))
    train()