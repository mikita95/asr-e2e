import argparse

import tensorflow as tf

import asr.nn.model as mb
from asr.models.params.modes import Mode

FLAGS = None


def sparse_to_labels(sparse_matrix):
    import asr.utils.data.examples.labels.handler as hn
    handler = hn.IndexerLabelsHandler(
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


def evaluate():
    """ Evaluate deepSpeech modelfor a number of steps."""
    import asr.models.ctc.input
    import asr.utils.data.examples.labels.handler as hn

    with tf.Graph().as_default() as graph:
        # Get feats and labels for deepSpeech.
        tot_batch_size = int(FLAGS['batch_size'])

        filenames, iterator = asr.models.ctc.input.inputs(batch_size=int(FLAGS['batch_size']),
                                                          num_epochs=None,
                                                          shuffle=bool(FLAGS['shuffle']))

        features, labels = iterator.get_next()

        # Build ops that computes the logits predictions from the
        # inference model.

        logits = mb.create_model(arch_type=FLAGS['model'],
                                 feature_input=features[0],
                                 seq_lengths=features[1],
                                 batch_size=int(FLAGS['batch_size']),
                                 mode=Mode.TRAIN,
                                 num_classes=hn.IndexerLabelsHandler(
                                     alphabet_file=FLAGS['alphabet_config_file']).get_alphabet_size() + 1,
                                 config_file=FLAGS['model_config_file'])

        output_log_prob = tf.nn.log_softmax(logits)
        decoder = tf.nn.ctc_beam_search_decoder
        predictions = decoder(output_log_prob, features[1])

        saver = tf.train.Saver()
        with tf.Session() as sess:

            # Initialize weights from checkpoint file.
            global_step = initialize_from_checkpoint(sess, saver)
            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS['test_record_path']]})

            inference(predictions, tf.deserialize_many_sparse(labels, dtype=tf.int32), sess, FLAGS['output_file'])


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
