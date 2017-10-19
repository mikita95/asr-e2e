import os
import re
from abc import ABC, abstractmethod
import tensorflow as tf
from collections import namedtuple

import asr.utils.data.examples.features.selector as fsb
import asr.utils.data.examples.labels.handler as lh


class Processor(ABC):
    def __init__(self, features_selector: fsb.FeatureSelector=None, labels_handler: lh.SentencesLabelsHandler=None):
        self.labels_handler = labels_handler
        self.features_selector = features_selector

    Example = namedtuple('Example', ['features_vector', 'label'])

    @abstractmethod
    def encode_example(self, parsed_example: Example) -> tf.train.Example:
        pass

    @abstractmethod
    def data_dir_handle(self, data_dir: str) -> list:
        pass

    @abstractmethod
    def parse_example(self, unparsed_example) -> Example:
        pass

    def write(self, data_dir, record_file_path, logging_freq=50):
        tf_writer = tf.python_io.TFRecordWriter(record_file_path)
        unparsed_examples = self.data_dir_handle(data_dir)

        for n, unparsed_ex in enumerate(unparsed_examples):
            parsed_example = self.parse_example(unparsed_example=unparsed_ex)
            encoded_example = self.encode_example(parsed_example)

            tf_writer.write(encoded_example)

            if n % logging_freq == 0:
                tf.logging.info("Processed: %d / %d" % (n, len(unparsed_examples)))

        tf_writer.close()

    @abstractmethod
    def decode_examples(self, filename_queue):
        pass

    def generate_input(self, tfrecords_path, shuffle: bool):
        filename_queue = tf.train.string_input_producer([tfrecords_path], shuffle=shuffle)
        return self.decode_examples(filename_queue=filename_queue)


class AudioDataProcessor(Processor):
    def __init__(self, config_file=None, features_selector: fsb.AudioFeatureSelector = None,
                 labels_handler: lh.SentencesLabelsHandler = None):

        if not (config_file is None):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file, encoding='utf8')

            self.features_settings = dict(config.items('FEATURES'))
            self.labels_settings = dict(config.items('LABELS'))

            if features_selector is None:
                self.features_selector = fsb.get_audio_feature_selector(self.features_settings['features_selector'])
            if labels_handler is None:
                self.labels_handler = lh.get_labels_handler(alphabet_file=self.labels_settings['alphabet_config'],
                                                            handler_name=self.labels_settings['labels_handler'])

        super().__init__(features_selector, labels_handler)

    def data_dir_handle(self, data_dir: str) -> list:
        """
        Expected data directory structure:
            data >
                number >
                    %format [wav] >
                        [file_name.%format]*
                    etc >
                        txt.done.data: [( file_name "phrase" )\n]*
        Args:
            data_dir (str): data root

        Returns:
            unparsed_examples (list(dict(str -> str)): list of dicts {'audio_file_path', 'label'}

        """

        def _parse_labels_file(labels_file_path):
            """Parses file with target labels. Expected the following file format: [( [label_name] "[target_text]" )\n]*
            Args:
                labels_file_path (str): path to the labels file

            Returns:
                result (dict(str -> str)): dictionary where key is the name of the label and the value is the target text

            """
            result = {}
            with open(labels_file_path, encoding="utf8") as f:
                tf.logging.debug("Parsing labels file " + labels_file_path)

                content = f.readlines()
                for x in content:
                    result[x.split(" ")[1]] = re.findall('"([^"]*)"', x)[0]

                tf.logging.debug("Found " + str(len(result)) + " labels in " + labels_file_path)

                return result

        unparsed_examples = []

        for data_subset_dir_name in os.listdir(data_dir):  # data/001
            subset_format_dir_path = os.path.join(data_dir, data_subset_dir_name, 'wav')  # data/001/wav
            tf.logging.debug("Going to dir " + subset_format_dir_path)

            label_file_path = os.path.join(data_dir, data_subset_dir_name, 'etc', 'txt.done.data')
            labels_of_current_subset = _parse_labels_file(label_file_path)

            for audio_file_name in os.listdir(subset_format_dir_path):
                feature_file_path = os.path.join(subset_format_dir_path, audio_file_name)

                #  append pair (path to the audio file, label text of the file)
                example_name = os.path.splitext(audio_file_name)[0]
                unparsed_examples.append({'audio_file_path': feature_file_path,
                                          'label': labels_of_current_subset[example_name]})

        return unparsed_examples

    def parse_example(self, unparsed_example: dict) -> Processor.Example:
        """Parses example of data

        Args:
            unparsed_example (dict(str -> str)): dictionary where key is the name of the label and the value is the target text

        Returns:
            example (Processor.Example): parsed example of data
        """
        features_vector = self.features_selector.get_audio_features_vector(file_path=unparsed_example['audio_file_path'],
                                                                                       settings=self.features_settings)
        label = self.labels_handler.encode(unparsed_example['label'])

        return Processor.Example(features_vector=features_vector,
                                 label=label)

    def encode_example(self, parsed_example: Processor.Example) -> tf.train.Example:
        """Serialize example of data to tensorflow string

        Args:
            parsed_example (Processor.Example): parsed example

        Returns:
            example (tf.train.Example): tensorflow representation of data

        """
        seq_length = parsed_example.features_vector[1]

        # Make context features
        seq_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_length]))
        context_features = tf.train.Features(feature={"seq_length": seq_length_feature, "label": parsed_example.label})

        ex = tf.train.SequenceExample(context=context_features,
                                      feature_lists=parsed_example.features_vector[0])

        return ex.SerializeToString()

    def decode_examples(self, filename_queue):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='writer', description='Script to write data')

    parser.add_argument('--writer_config',
                        type=str,
                        help='Path to writer config file')

    parser.add_argument('--data_dir',
                        type=str)

    parser.add_argument('--record_path',
                        type=str)

    ARGS, unparsed = parser.parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    writer = DataProcessor(config_file=ARGS.writer_config)
    writer.write(data_dir=ARGS.data_dir,
                 record_file_path=ARGS.record_path)
