import argparse
import utils.data_processor as dp
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
import models.models as md
import os

FLAGS = None


def run():
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

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

    target_ixs = tf.placeholder(tf.int64)
    target_values = tf.placeholder(tf.int32)
    target_shape = tf.placeholder(tf.int64)
    target = tf.SparseTensor(target_ixs, target_values, target_shape)

    # TODO: dims
    input = tf.placeholder(tf.float32, [None], name='fingerprint_input')

    output = md.create_model(FLAGS.model, input, None, FLAGS.mode)
    seq_lengths = tf.placeholder(tf.int32, shape=FLAGS.batch_size)
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('ctc'):
        ctc_mean = tf.reduce_mean(ctc.ctc_loss(target, output, seq_lengths))

   # tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(ctc_mean)

    logits_max_test = tf.slice(tf.argmax(output, 2), [0, 0], [seq_lengths[0], 1])

    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(output, seq_lengths)[0][0])
    evaluation_step = tf.reduce_sum(
        tf.edit_distance(predictions, target, normalize=False)) / tf.to_float(
        tf.size(target.values))

    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()
    start_step = 1
    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break
        # Pull the audio samples we'll use for training.

        train_data = dp.load_batched_data(FLAGS.data_path, FLAGS.batch_size, FLAGS.mode)

       # train_fingerprints, train_ground_truth = audio_processor.get_data(
       #     FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
       #    FLAGS.background_volume, time_shift_samples, 'training', sess)
        # Run the graph with this batch of training data.
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, evaluation_step, ctc_mean, train_step,
                increment_global_step
            ],
            feed_dict={
            # TODO: map data
            #    fingerprint_input: train_fingerprints,
            #    ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
            })
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))
        is_last_step = (training_step == training_steps_max)
        # TODO: implement validation
        # if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
        #     set_size = audio_processor.set_size('validation')
        #     total_accuracy = 0
        #     total_conf_matrix = None
        #     for i in xrange(0, set_size, FLAGS.batch_size):
        #         validation_fingerprints, validation_ground_truth = (
        #             audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
        #                                      0.0, 0, 'validation', sess))
        #         # Run a validation step and capture training summaries for TensorBoard
        #         # with the `merged` op.
        #         validation_summary, validation_accuracy, conf_matrix = sess.run(
        #             [merged_summaries, evaluation_step, confusion_matrix],
        #             feed_dict={
        #                 fingerprint_input: validation_fingerprints,
        #                 ground_truth_input: validation_ground_truth,
        #                 dropout_prob: 1.0
        #             })
        #         validation_writer.add_summary(validation_summary, training_step)
        #         batch_size = min(FLAGS.batch_size, set_size - i)
        #         total_accuracy += (validation_accuracy * batch_size) / set_size
        #         if total_conf_matrix is None:
        #             total_conf_matrix = conf_matrix
        #         else:
        #             total_conf_matrix += conf_matrix
        #     tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        #     tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
        #                     (training_step, total_accuracy * 100, set_size))

        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            checkpoint_path = os.path.join(FLAGS.train_dir,
                                           FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    # TODO: implement testing
    # set_size = audio_processor.set_size('testing')
    # tf.logging.info('set_size=%d', set_size)
    # total_accuracy = 0
    # total_conf_matrix = None
    # for i in xrange(0, set_size, FLAGS.batch_size):
    #     test_fingerprints, test_ground_truth = audio_processor.get_data(
    #         FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    #     test_accuracy, conf_matrix = sess.run(
    #         [evaluation_step, confusion_matrix],
    #         feed_dict={
    #             fingerprint_input: test_fingerprints,
    #             ground_truth_input: test_ground_truth,
    #             dropout_prob: 1.0
    #         })
    #     batch_size = min(FLAGS.batch_size, set_size - i)
    #     total_accuracy += (test_accuracy * batch_size) / set_size
    #     if total_conf_matrix is None:
    #         total_conf_matrix = conf_matrix
    #     else:
    #         total_conf_matrix += conf_matrix
    # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    # tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
    #                                                          set_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and test models')

    parser.add_argument('--mode',
                        type=str,
                        help='Running mode',
                        choices=['train', 'test'],
                        default='train')

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to data dir')

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
