# coding=utf-8
import os.path
import glob
import tensorflow as tf

ALPHABET = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
NUM_CLASSES = len(ALPHABET) + 1  # Additional class for blank
CHAR_TO_IX = {ch: i for (i, ch) in enumerate(ALPHABET)}


def normalize_label_text(label_text):
    return ' '.join(label_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', ''). \
        replace("'", '').replace('!', '').replace('-', '')


def convert_label_to_ctc_format(label_text):
    import numpy as np
    original = normalize_label_text(label_text)

    label = np.asarray([CHAR_TO_IX[c] for c in original if c in CHAR_TO_IX])

    return label


def _generate_feats_and_label_batch(filename_queue, batch_size):
    """Construct a queued batch of spectral features and transcriptions.
    Args:
      filename_queue: queue of filenames to read data from.
      batch_size: Number of utterances per batch.
    Returns:
      feats: mfccs. 4D tensor of [batch_size, height, width, 3] size.
      labels: transcripts. List of length batch_size.
      seq_lens: Sequence Lengths. List of length batch_size.
    """

    # Define how to parse the example
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "seq_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label": tf.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        "features": tf.FixedLenSequenceFeature([None, ], dtype=tf.float32)
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