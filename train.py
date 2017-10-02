# coding=utf-8
import argparse
import sys
import time
from datetime import datetime

import tensorflow as tf
import nn.models_builder as mb
import examples_writer
import re

FLAGS = None

TOWER_NAME = 'tower'


def loss_f(logits, labels, seq_lens):
    """Compute mean CTC Loss.
        Add summary for "Loss" and "Loss/avg".
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]
          seq_lens: Length of each utterance for ctc cost computation.
        Returns:
          Loss tensor of type float.
        """
    # Calculate the average ctc loss across the batch.
    ctc_loss = tf.nn.ctc_loss(inputs=tf.cast(logits, tf.float32),
                              labels=labels, sequence_length=seq_lens)
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
    tf.add_to_collection('losses', ctc_loss_mean)

    # The total loss is defined as the cross entropy loss plus all
    # of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, feats, labels, seq_lens):
    """Calculate the total loss on a single tower running the deepSpeech model.
    This function builds the graph for computing the loss per tower(GPU).
    ARGS:
      scope: unique prefix string identifying the
             deepSpeech tower, e.g. 'tower_0'
      feats: Tensor of shape BxFxT representing the
             audio features (mfccs or spectrogram).
      labels: sparse tensor holding labels of each utterance.
      seq_lens: tensor of shape [batch_size] holding
              the sequence length per input utterance.
    Returns:
       Tensor of shape [batch_size] containing
       the total loss for a batch of data
    """

    # Build inference Graph.
    logits = mb.create_model(arch_type=FLAGS.model,
                             feature_input=feats,
                             seq_lengths=seq_lens,
                             mode='train',
                             num_classes=len(examples_writer.ALPHABET) + 1)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    strided_seq_lens = tf.div(seq_lens)
    _ = loss_f(logits, labels, strided_seq_lens)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for loss in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a
        # multi-GPU training session. This helps the clarity
        # of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '',
                           loss.op.name)
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.summary.scalar(loss_name + '(raw)', loss)
        tf.summary.scalar(loss_name, loss_averages.average(loss))

    # Without this loss_averages_op would never run
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the
       gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for each_grad, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(each_grad, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # The variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        weights = grad_and_vars[0][1]
        grad_and_var = (grad, weights)
        average_grads.append(grad_and_var)
    return average_grads


def set_learning_rate():
    """ Set up learning rate schedule """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed * ARGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = FLAGS.num_batches_per_epoch
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        FLAGS.initial_lr,
        global_step,
        decay_steps,
        FLAGS.lr_decay_factor,
        staircase=True)

    return learning_rate, global_step


def fetch_data():
    import models.ctc_input
    """ Fetch features, labels and sequence_lengths from a common queue."""

    tot_batch_size = FLAGS.batch_sizes
    feats, labels, seq_lens = models.ctc_input.inputs(
                                                tfrecords_path=FLAGS.data_dir,
                                                batch_size=tot_batch_size,
                                                shuffle=FLAGS.shuffle)

    # Split features and labels and sequence lengths for each tower
    split_feats = tf.split(0, 1, feats)
    split_labels = tf.sparse_split(0, 1, labels)
    split_seq_lens = tf.split(0, 1, seq_lens)

    return split_feats, split_labels, split_seq_lens


def get_loss_grads(data, optimizer):
    """ Set up loss and gradient ops.
    Add summaries to trainable variables """

    # Calculate the gradients for each model tower.
    global summaries, loss
    [feats, labels, seq_lens] = data
    tower_grads = []
    for i in range(1):
        with tf.device('/gpu:%d' % i):
            name_scope = '%s_%d' % (TOWER_NAME, i)
            with tf.name_scope(name_scope) as scope:
                # Calculate the loss for one tower of the deepSpeech model.
                # This function constructs the entire deepSpeech model
                # but shares the variables across all towers.
                loss = tower_loss(scope, feats[i], labels[i], seq_lens[i])

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                # Calculate the gradients for the batch of
                # data on this tower.
                grads_and_vars = optimizer.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads_and_vars)

    return loss, tower_grads, summaries


def run_train_loop(sess, operations, saver):
    import numpy as np
    import os
    """ Train the model for required number of steps."""
    (train_op, loss_op, summary_op) = operations
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # Evaluate the ops for max_steps
    for step in range(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss_op])
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # Print progress periodically.
        if step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / duration
            format_str = ('%s: step %d, '
                          'loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, duration))

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


def add_summaries(summaries, learning_rate, grads):
    """ Add summary ops"""

    # Track quantities for Tensorboard display
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(
                tf.summary.histogram(var.op.name +
                                     '/gradients', grad))
    # Add histograms for trainable variables.
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
    with tf.Graph().as_default(), tf.device('/cpu'):

        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Fetch a batch worth of data for each tower
        data = fetch_data()

        # Construct loss and gradient ops
        loss_op, tower_grads, summaries = get_loss_grads(data, optimizer)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads,
                                                      global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Build summary op
        summary_op = add_summaries(summaries, learning_rate, grads)

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
        run_train_loop(sess, (train_op, loss_op, summary_op), saver)


if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    parser = argparse.ArgumentParser(prog='trainer', description='Script to train and test nn')

    num_gpus = len([x for x in device_lib.list_local_devices()
                    if x.device_type == "GPU"])

    parser.add_argument('--data_path',
                        type=str,
                        help='Path to data dir')

    parser.add_argument('--val_path',
                        type=str,
                        help='Path to validation data dir')

    parser.add_argument('--model',
                        help='Name of neural network model',
                        type=str)


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

    parser.add_argument('--num_batches_per_epoch', type=int, default=100)

    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Number of batches to run')
    parser.add_argument('--num_gpus', type=int, default=num_gpus,
                        help='How many GPUs to use')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of inputs to process in a batch per GPU')
    parser.add_argument('--temporal_stride', type=int, default=2,
                        help='Stride along time')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle', dest='shuffle',
                                action='store_true')
    feature_parser.add_argument('--no-shuffle', dest='shuffle',
                                action='store_false')
    parser.set_defaults(shuffle=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--use_fp16', dest='use_fp16',
                                action='store_true')
    feature_parser.add_argument('--use_fp32', dest='use_fp16',
                                action='store_false')
    parser.set_defaults(use_fp16=False)

    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Continue training from checkpoint file')

    parser.add_argument('--initial_lr', type=float, default=0.00001,
                        help='Initial learning rate for training')

    parser.add_argument('--moving_avg_decay', type=float, default=0.9999,
                        help='Decay to use for the moving average of weights')
    parser.add_argument('--num_epochs_per_decay', type=int, default=5,
                        help='Epochs after which learning rate decays')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9,
                        help='Learning rate decay factor')

    FLAGS, unparsed = parser.parse_known_args()

