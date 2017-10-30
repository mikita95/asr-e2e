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
        "features": tf.FixedLenSequenceFeature([18, ], dtype=tf.float32)
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


def parser(record):
    context_features = {
        "seq_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label": tf.VarLenFeature(dtype=tf.int64)
    }

    sequence_features = {
        "features": tf.FixedLenSequenceFeature([18, ], dtype=tf.float32)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    label = context_parsed['label']
    #label = tf.sparse_to_dense(sparse_indices=label.indices,
    #                           sparse_values=label.values,
    #                           output_shape=label.dense_shape)

    return sequence_parsed['features'], tf.serialize_sparse(tf.cast(label, tf.int32)), tf.cast(context_parsed['seq_length'], tf.int32)


def inputs(batch_size, num_epochs, shuffle=False):
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.contrib.data.TFRecordDataset(filenames).map(parser).repeat(num_epochs)

    labels_dataset = dataset.map(lambda f, l, s: l).batch(batch_size=batch_size)
    padded_dataset = dataset.map(lambda f, l, s: (f, s)).padded_batch(batch_size=batch_size,
                                                                      padded_shapes=([-1, 18], []))

    dataset = tf.contrib.data.Dataset.zip((padded_dataset, labels_dataset)).shuffle(buffer_size=200)


    return filenames, dataset.make_initializable_iterator()
