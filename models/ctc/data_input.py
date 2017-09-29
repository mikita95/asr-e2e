import os
import re

import numpy as np
import tensorflow as tf

from feature_selector.feature_selector_builder import get_feature_selector

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('а') - 1  # 0 is reserved to space


def normalize_label_text(label_text):
    return ' '.join(label_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', ''). \
        replace("'", '').replace('!', '').replace('-', '')


def convert_label_to_ctc_format(label_text):
    original = normalize_label_text(label_text)

    labels = original.replace(' ', '  ')
    labels = labels.split(' ')  # list of dims [words + blanks]

    # Adding blank label
    labels = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in labels])  # array [number_of_characters + blanks]

    # Transform char into index, array [number_of_characters + blanks]
    label = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in labels])
    return label


def read_data(examples_queue, feature_settings, rate=16000, channel_count=1, mode='mfcc', format='wav'):
    class DataRecord(object):
        def __init__(self):
            self.label = None
            self.feature_vector = None

    result = DataRecord()
    feature_selector = get_feature_selector(mode)
    feature_vector = feature_selector.get_feature_vector(file_path=examples_queue[0],
                                                         feature_settings=feature_settings,
                                                         samples_per_second=rate,
                                                         channel_count=channel_count,
                                                         file_format=format)
    result.feature_vector = feature_vector
    result.label = convert_label_to_ctc_format(examples_queue[1])

    return result


def _generate_audio_and_label_batch(feature_vector, label, min_queue_examples,
                                    batch_size):
    num_preprocess_threads = 4

    feature_vectors, label_batch = tf.train.shuffle_batch(
            [feature_vector, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    return feature_vectors, tf.reshape(label_batch, [batch_size])


def parse_labels_file(file_path):
    """
    Parses file with target labels. Expected the following file format: [( [label_name] "[target_text]" )\n]*
    Args:
        file_path: path to the labels file

    Returns: dictionary where key is the name of the label and the value is the target text

    """
    result = {}
    with open(file_path, encoding="utf8") as f:
        content = f.readlines()
        for x in content:
            result[x.split(" ")[1]] = re.findall('"([^"]*)"', x)[0]
        return result


def data_dir_handle(data_dir, format='wav'):
    audio_file_paths = []
    labels = []

    for data_subset_dir_name in os.listdir(data_dir):  # data/001
        subset_format_dir_path = os.path.join(data_dir, data_subset_dir_name, format)  # data/001/wav
        tf.logging.info("Going to dir " + subset_format_dir_path)

        label_file_path = os.path.join(data_dir, data_subset_dir_name, 'etc', 'txt.done.data')
        labels_of_current_subset = parse_labels_file(label_file_path)

        for audio_file_name in os.listdir(subset_format_dir_path):
            audio_file_path = os.path.join(subset_format_dir_path, audio_file_name)

            #  append pair (path to the audio file, label text of the file)
            audio_file_paths.append(audio_file_path)
            example_name = os.path.splitext(audio_file_name)[0]
            labels.append(labels_of_current_subset[example_name])

    return audio_file_paths, labels


def inputs(data_dir, max_examples_per_epoch, batch_size, feature_settings, rate=16000, channel_count=1, mode='mfcc', format='wav'):
    audio_file_paths, labels = data_dir_handle(data_dir, format)

    # Create a queue that produces the filenames to read.
    examples_queue = tf.train.slice_input_producer([audio_file_paths, labels], shuffle=True)

    # Read examples from files in the filename queue.
    read_input = read_data(examples_queue,
                           feature_settings=feature_settings,
                           rate=rate,
                           channel_count=channel_count,
                           mode=mode,
                           format=format)

    feature_vector = read_input.feature_vector
    label = read_input.label

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(max_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_audio_and_label_batch(feature_vector, label, min_queue_examples, batch_size)
