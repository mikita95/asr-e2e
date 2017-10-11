# coding=utf-8
import tensorflow as tf


def _generate_feats_and_label_batch(filename_queue, batch_size):
    """Construct a queued batch of spectral features and transcriptions.
    Args:
      filename_queue: queue of filenames to read data from.
      batch_size: Number of utterances per batch.
    Returns:
      feats: mfccs. 4D tensor of [B, T, S]
      labels: SparseTensor of dense shape [B, L]
      seq_lens: Sequence Lengths. List of length B.
    """

    # Define how to parse the example
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "seq_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label": tf.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        "features": tf.FixedLenSequenceFeature([13, ], dtype=tf.float32)
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Generate a batch worth of examples after bucketing
    seq_len, (feats, labels) = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.cast(context_parsed['seq_length'], tf.int32),
        tensors=[sequence_parsed['features'], context_parsed['label']],
        batch_size=batch_size,
        bucket_boundaries=list(range(100, 1900, 100)),
        allow_smaller_final_batch=True,
        num_threads=4,
        dynamic_pad=True)

    return feats, tf.cast(labels, tf.int32), seq_len


def inputs(tfrecords_path, batch_size, shuffle=False):
    """Construct input for fordspeech evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the fordspeech data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of
              [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([tfrecords_path], shuffle=shuffle)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_feats_and_label_batch(filename_queue, batch_size)


if __name__ == '__main__':
    with tf.Graph().as_default():
        feats, labels, seq_len = inputs("C:\\Users\\Nikita_Markovnikov\\Downloads\\train_records.tf", 8)
        print("hello")
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        tf.train.start_queue_runners(sess)
        with sess.as_default():
            print(feats.eval().shape)
            print(labels.eval().dense_shape)
            print(seq_len.eval())

