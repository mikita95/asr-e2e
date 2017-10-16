import argparse

import tensorflow as tf

import asr.nn.model as mb
from asr.models.params.modes import Mode

FLAGS = None


def sparse_to_labels(sparse_matrix):
    import asr.models.ctc.labels.handler as hn
    handler = hn.CTCLabelsHandler(
        alphabet_file=FLAGS['alphabet_config_file'])
    """ Convert index based transcripts to strings"""
    results = [''] * sparse_matrix.dense_shape[0]
    for i, val in enumerate(sparse_matrix.values.tolist()):
        results[sparse_matrix.indices[i, 0]] += handler.decode([val])
    return results


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


def inference(predictions_op, true_labels_op, sess, output_file):
    """ Perform inference per batch on pre-trained model.
    This function performs inference and computes the CER per ut+-terance.
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


def eval_once(saver, predictions_op, true_labels_op):
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

            inference(predictions_op, true_labels_op, sess, FLAGS['output_file'])

        except Exception as exc:  # pylint: disable=broad-except
            coord.request_stop(exc)

        # Close threads
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """ Evaluate deepSpeech modelfor a number of steps."""
    import asr.models.ctc.input
    import asr.models.ctc.labels.handler as hn

    with tf.Graph().as_default() as graph:
        # Get feats and labels for deepSpeech.
        tot_batch_size = int(FLAGS['batch_size'])

        feats, labels, seq_lens = asr.models.ctc.input.inputs(
            tfrecords_path=FLAGS['test_record_path'],
            batch_size=tot_batch_size,
            shuffle=bool(FLAGS['shuffle']))

        # Build ops that computes the logits predictions from the
        # inference model.

        logits = mb.create_model(arch_type=FLAGS['model'],
                                 feature_input=feats,
                                 seq_lengths=seq_lens,
                                 batch_size=int(FLAGS['batch_size']),
                                 mode=Mode.TRAIN,
                                 num_classes=hn.CTCLabelsHandler(
                                     alphabet_file=FLAGS['alphabet_config_file']).get_alphabet_size() + 1,
                                 config_file=FLAGS['model_config_file'])

        # Calculate predictions.
        output_log_prob = tf.nn.log_softmax(logits)
        decoder = tf.nn.ctc_beam_search_decoder

        predictions = decoder(output_log_prob, seq_lens)

        saver = tf.train.Saver()

        eval_once(saver, predictions, labels)


def main():
    evaluate()


if __name__ == '__main__':
    """ Parses command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_config',
                        type=str,
                        help='Path to test config file')

    ARGS, unparsed = parser.parse_known_args()
    import configparser

    config = configparser.ConfigParser()
    config.read(ARGS.test_config, encoding='utf8')
    FLAGS = dict(config.items('CONFIGS'))

    main()
