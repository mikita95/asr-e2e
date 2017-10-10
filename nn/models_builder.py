from nn.lstm_arch.lstm import LSTM
import tensorflow as tf


def create_model(arch_type, feature_input, seq_lengths, mode, num_classes, batch_size=None, config_file=None):
    if arch_type == 'lstm_arch':
        return LSTM(feature_input=feature_input,
                    seq_lengths=seq_lengths,
                    num_classes=num_classes,
                    mode=mode,
                    batch_size=batch_size,
                    config_file='lstm_config.ini' if config_file is None else config_file).build_graph()


def load_variables_from_checkpoint(sess, start_checkpoint):
    """
    Utility function to centralize checkpoint restoration.
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
