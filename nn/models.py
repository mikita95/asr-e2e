from nn.lstm import LSTM
import tensorflow as tf


def create_model(arch_type, feature_input, seq_lengths, mode, num_classes, settings=None):
    if arch_type == 'lstm':
        return LSTM(feature_input, seq_lengths, mode, num_classes, settings).build_graph()


def load_variables_from_checkpoint(sess, start_checkpoint):
    """
    Utility function to centralize checkpoint restoration.
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
