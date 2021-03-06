# coding=utf-8
import argparse
import time
from datetime import datetime

import tensorflow as tf

import asr.nn.model as mb
from asr.models.params.modes import Mode
import os

FLAGS = None


def get_loss(feats, labels, seq_lens, mode=Mode.TRAIN):
    import asr.utils.data.examples.labels.handler as hn

    logits = mb.create_model(arch_type=FLAGS['model'],
                             feature_input=feats,
                             seq_lengths=seq_lens,
                             batch_size=int(FLAGS['batch_size']),
                             mode=mode,
                             num_classes=hn.IndexerLabelsHandler(
                                 alphabet_file=FLAGS['alphabet_config_file']).get_alphabet_size() + 1,
                             config_file=FLAGS['model_config_file'])

    ctc_loss = tf.nn.ctc_loss(inputs=logits,
                              labels=tf.deserialize_many_sparse(labels, dtype=tf.int32),
                              sequence_length=seq_lens)

    ctc_loss_mean = tf.reduce_mean(ctc_loss)

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                               sequence_length=seq_lens)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), tf.deserialize_many_sparse(labels, dtype=tf.int32)))

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
    import asr.models.ctc.input
    import numpy as np

    with tf.Graph().as_default(), tf.device('/cpu'):
        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        """ Fetch features, labels and sequence_lengths from a common queue."""
        filenames, iterator = asr.models.ctc.input.inputs(batch_size=int(FLAGS['batch_size']),
                                                          num_epochs=int(FLAGS['max_steps']),
                                                          shuffle=bool(FLAGS['shuffle']))

        features, labels = iterator.get_next()

        # Create an optimizer that performs gradient descent.
        optimizer = opt.get(FLAGS['optimizer'])(learning_rate)

        train_loss_op, train_ler_op = get_loss(feats=features[0],
                                               labels=labels,
                                               seq_lens=features[1],
                                               mode=Mode.TRAIN)

        train_op = optimizer.minimize(train_loss_op, global_step=global_step)

        # Create summaries for TRAIN
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        train_summary_op = add_summaries(summaries, learning_rate, train_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=bool(FLAGS['log_device_placement'])))

        # Initialize vars.
        sess.run(tf.initialize_all_variables())

        summary_writer = tf.summary.FileWriter(FLAGS['train_dir'], sess.graph)
        all_step = 0
        for step in range(int(FLAGS['max_steps'])):
            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS['train_record_path']]})
            batch_num = 0

            try:
                for i in range(0, int(FLAGS['num_batches_per_epoch'])):
                    start_time = time.time()

                    sess.run([features, labels])
                    train_loss_value, _, train_ler_value = sess.run([train_loss_op, train_op, train_ler_op])

                    assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'
                    duration = time.time() - start_time

                    examples_per_sec = int(FLAGS['batch_size']) / duration
                    format_str = ('%s: Training: step %d, batch %d, '
                                  'ler = %.2f, '
                                  'loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    tf.logging.info(
                        format_str % (datetime.now(), step, batch_num, train_ler_value, train_loss_value, examples_per_sec,
                                      duration))

                    if all_step % int(FLAGS['summaries_interval']) == 0:
                        summary_writer.add_summary(sess.run(train_summary_op), all_step)

                    # Save the model checkpoint periodically.
                    if all_step % int(FLAGS['model_save_period']) == 0:
                        checkpoint_path = os.path.join(FLAGS['train_dir'], 'model.ckpt')
                        saver.save(sess=sess,
                                   save_path=checkpoint_path,
                                   global_step=all_step)
                    batch_num += 1
                    all_step += 1

            finally:
                print("End of training epoch " + str(step))

            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS['val_record_path']]})
            av_val_ler = 0
            num_val_batches = 0

            try:
                for i in range(0, int(FLAGS['num_batches_per_val'])):
                    sess.run([features, labels])
                    train_loss_value, train_ler_value = sess.run([train_loss_op, train_ler_op])

                    assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'

                    av_val_ler += train_ler_value
                    num_val_batches += 1

            finally:
                format_str = ('Validation: step %d, '
                              'ler = %.2f')
                tf.logging.info(
                    format_str % (step, av_val_ler / num_val_batches))
        if all_step % int(FLAGS['model_save_period']) == 0:
            checkpoint_path = os.path.join(FLAGS['train_dir'], 'model.ckpt')
            saver.save(sess=sess,
                       save_path=checkpoint_path,
                       global_step=all_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainer',
                                     description='Script to train nn')

    parser.add_argument('--train_config',
                        type=str,
                        help='Path to train config file')

    ARGS, _ = parser.parse_known_args()

    import configparser
    config = configparser.ConfigParser()
    config.read(ARGS.train_config, encoding='utf8')
    FLAGS = dict(config.items('CONFIGS'))

    tf.logging.set_verbosity(tf.logging.INFO)

    train()
