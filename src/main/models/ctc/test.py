import argparse

import tensorflow as tf

from src.main import nn as mb
from src.main.utils import writer

# Note this definition must match the ALPHABET chosen in
# preprocess_Librispeech.py
ALPHABET = writer.ALPHABET
IX_TO_CHAR = {i: ch for (i, ch) in enumerate(ALPHABET)}

FLAGS = None


def sparse_to_labels(sparse_matrix):
    """ Convert index based transcripts to strings"""
    results = [''] * sparse_matrix.dense_shape[0]
    for i, val in enumerate(sparse_matrix.values.tolist()):
        results[sparse_matrix.indices[i, 0]] += IX_TO_CHAR[val]
    return results


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""

    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
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


def inference(predictions_op, true_labels_op, sess, output_file):
    """ Perform inference per batch on pre-trained model.
    This function performs inference and computes the CER per utterance.
    Args:
        predictions_op: Prediction op
        true_labels_op: True Labels op
        display: print sample predictions if True
        sess: default session to evaluate the ops.
    Returns:
        char_err_rate: list of CER per utterance.
    """
    char_err_rate = []
    # Perform inference of batch worth of data at a time.
    [predictions, true_labels] = sess.run([predictions_op, true_labels_op])

    pred_label = sparse_to_labels(predictions[0][0])
    actual_label = sparse_to_labels(true_labels)
    with open(output_file, 'w') as f:
        for (label, pred) in zip(actual_label, pred_label):
            f.write('Predicted: ' + str(pred) + '\n')
            f.write('Expected: ' + str(label) + '\n\n')

    return char_err_rate


def eval_once(saver, summary_writer, predictions_op, summary_op,
              true_labels_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      predictions_ops: Op to compute predictions.
      summary_op: Summary op.
    """
    with tf.Session() as sess:

        # Initialize weights from checkpoint file.
        global_step = initialize_from_checkpoint(sess, saver)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for queue_runners in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue_runners.create_threads(sess, coord=coord,
                                                            daemon=True,
                                                            start=True))
            # Only using a subset of the training data

            inference(predictions_op, true_labels_op, sess, FLAGS.output_file)

            # Add summary ops
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)
        except Exception as exc:  # pylint: disable=broad-except
            coord.request_stop(exc)

        # Close threads
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """ Evaluate deepSpeech modelfor a number of steps."""

    with tf.Graph().as_default() as graph:
        # Get feats and labels for deepSpeech.
        tot_batch_size = FLAGS.batch_size

        feats, labels, seq_lens = src.models.ctc.input.inputs(
            tfrecords_path=FLAGS.record_path,
            batch_size=tot_batch_size,
            shuffle=FLAGS.shuffle)

        # Build ops that computes the logits predictions from the
        # inference model.
        FLAGS.keep_prob = 1.0  # Disable dropout during testing.
        logits = mb.create_model(arch_type=FLAGS.model,
                                 feature_input=feats,
                                 seq_lengths=seq_lens,
                                 mode='train',
                                 num_classes=len(writer.ALPHABET) + 1,
                                 settings=FLAGS)

        # Calculate predictions.
        output_log_prob = tf.nn.log_softmax(logits)
        decoder = tf.nn.ctc_beam_search_decoder
        strided_seq_lens = tf.div(seq_lens, FLAGS.temporal_stride)
        predictions = decoder(output_log_prob, strided_seq_lens)

        print(predictions)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_avg_decay)

        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph)

        eval_once(saver, summary_writer, predictions, summary_op, labels)


def main():
    """
    Create eval directory and perform inference on checkpointed model.
    """
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    """ Parses command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--record_path',
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

    parser.add_argument('--output_file', type=str)
    parser.add_argument('--eval_dir', type=str,
                        default='../models/librispeech/eval',
                        help='Directory to write event logs')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='../models/librispeech/train',
                        help='Directory where to read model checkpoints.')
    parser.add_argument('--eval_data', type=str, default='val',
                        help="Either 'test' or 'val' or 'train' ")

    parser.add_argument('--eval_interval_secs', type=int, default=60 * 5,
                        help='How often to run the eval')
    parser.add_argument('--data_dir', type=str,
                        default='../data/librispeech/processed/',
                        help='Path to the deepSpeech data directory')
    parser.add_argument('--run_once', type=bool, default=False,
                        help='Whether to run eval only once')
    parser.add_argument('--num_batches_per_epoch', type=int, default=100)

    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Number of batches to run')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='How many GPUs to use')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')

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
    args = parser.parse_args()

    FLAGS, unparsed = parser.parse_known_args()
    main()
