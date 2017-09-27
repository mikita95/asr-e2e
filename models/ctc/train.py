# coding=utf-8
import argparse
import utils.data_processor as dp
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
import nn.models as md
import os

FLAGS = None

# Accounting the 0th index + space + blank label = 28 characters
num_classes = ord('я') - ord('а') + 1 + 1 + 1


def run():
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    session = tf.InteractiveSession()

    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))

    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32, name='targets')

    feature_number = tf.placeholder(tf.int32, name='feature_number')  # e.g. number of cepstrals in mfcc

    # [max_time_steps x batch_size x feature_number]
    inputs = tf.placeholder(tf.float32, [None, None, feature_number], name='inputs')

    # Lengths of the audio sequences in frames, array [batch_size]
    seq_lengths = tf.placeholder(tf.int32, shape=FLAGS.batch_size, name='seq_lengths')

    # TODO: DIM?
    logits = md.create_model(arch_type=FLAGS.model,
                             feature_input=inputs,
                             settings=None,
                             mode=FLAGS.mode)


    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('ctc'):
        cost = tf.reduce_mean(ctc.ctc_loss(targets, logits, seq_lengths))
        tf.summary.scalar('ctc_cost', cost)

    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cost)

    decoded, log_prob = ctc.ctc_beam_search_decoder(logits, seq_lengths)

    evaluation_step = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))  # label error rate
    tf.summary.scalar('ler', evaluation_step)

    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         session.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        # TODO
      #  nn.load_variables_from_checkpoint(session, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=session)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(session.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')


    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):
        train_cost = train_ler = 0
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break
        # Pull the audio samples we'll use for training.

        train_data = dp.load_batched_data(FLAGS.data_path, FLAGS.batch_size, FLAGS.mode)
        batch_number = 0
        for batch in train_data:
            train_inputs, train_targets, train_seq_len, original = dp.handle_batch(batch)

            feed = {feature_number: train_inputs[0].shape[0],
                    inputs: train_inputs,
                    targets: train_targets,
                    learning_rate_input: learning_rate_value,
                    seq_lengths: train_seq_len}

            train_summary, train_ler, ctc_cost, _, _ = session.run(
                [merged_summaries, evaluation_step, cost, optimizer, increment_global_step], feed)

            train_writer.add_summary(train_summary, training_step)

            tf.logging.info('Batch #%d: rate %f, LER %.1f%%, CTC cost %f' %
                            (batch_number, learning_rate_value, train_ler * 100, ctc_cost))
            batch_number += 1

        val_data = dp.load_batched_data(FLAGS.val_path, FLAGS.batch_size, FLAGS.mode)
        total_accuracy = 0
        for batch in val_data:
            val_inputs, val_targets, val_seq_len, original = dp.handle_batch(batch)

            feed = {feature_number: val_inputs[0].shape[0],
                    inputs: val_inputs,
                    targets: val_targets,
                    seq_lengths: val_seq_len}

            validation_summary, val_ler = session.run(
                [merged_summaries, evaluation_step], feed)

            validation_writer.add_summary(validation_summary, training_step)
            total_accuracy += val_ler

        tf.logging.info('Step %d: Validation LER = %.1f%%' %
                       (training_step, total_accuracy * 100))

        # Save the model checkpoint periodically.
        if training_step % FLAGS.save_step_interval == 0 or training_step == training_steps_max:
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(session, checkpoint_path, global_step=training_step)


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
    run()
