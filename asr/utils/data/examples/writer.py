import os
import re

import tensorflow as tf

import asr.utils.data.examples.features.selector as fsb
import asr.utils.data.examples.labels.handler as lh


class Writer(object):
    def __init__(self, config_file=None, features_selector: fsb.FeatureSelector=None, labels_handler: lh.LabelsHandler=None):

        if not (config_file is None):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file, encoding='utf8')

            self.features_settings = dict(config.items('FEATURES'))
            self.labels_settings = dict(config.items('LABELS'))

        self.features_selector = features_selector
        self.labels_handler = labels_handler

        if features_selector is None:
            self.features_selector = fsb.get_feature_selector(self.features_settings['features_selector'])
        if labels_handler is None:
            self.labels_handler = lh.get_labels_handler(alphabet_file=self.labels_settings['alphabet_config'],
                                                        handler_name=self.labels_settings['labels_handler'])

    def _parse_labels_file(self, file_path):
        """
        Parses file with target labels. Expected the following file format: [( [label_name] "[target_text]" )\n]*
        Args:
            file_path: path to the labels file

        Returns: dictionary where key is the name of the label and the value is the target text

        """
        result = {}
        with open(file_path, encoding="utf8") as f:
            tf.logging.debug("Parsing labels file " + file_path)

            content = f.readlines()
            for x in content:
                result[x.split(" ")[1]] = re.findall('"([^"]*)"', x)[0]

            tf.logging.debug("Found " + str(len(result)) + " labels in " + file_path)

            return result

    def data_dir_handle(self, data_dir, audio_format='wav'):
        """
        Expected data directory structure:
            data >
                number >
                    %format [wav, mp3] >
                        [file_name.%format]*
                    etc >
                        txt.done.data: [( file_name "phrase" )\n]*
        Args:
            data_dir: data root
            audio_format: name of a directory corresponding audio format

        Returns: list of dicts {'audio_file_path', 'label'}

        """
        examples = []

        for data_subset_dir_name in os.listdir(data_dir):  # data/001
            subset_format_dir_path = os.path.join(data_dir, data_subset_dir_name, audio_format)  # data/001/wav
            tf.logging.debug("Going to dir " + subset_format_dir_path)

            label_file_path = os.path.join(data_dir, data_subset_dir_name, 'etc', 'txt.done.data')
            labels_of_current_subset = self._parse_labels_file(label_file_path)

            for audio_file_name in os.listdir(subset_format_dir_path):
                feature_file_path = os.path.join(subset_format_dir_path, audio_file_name)

                #  append pair (path to the audio file, label text of the file)
                example_name = os.path.splitext(audio_file_name)[0]
                examples.append({'audio_file_path': feature_file_path,
                                 'label': labels_of_current_subset[example_name]})

        return examples

    def encode_sequence_example(self, sequence, label):
        """
        Makes sequence example
        Args:
            sequence: [seq_length * N] numpy array
            label: string represents label
        Returns:
            Serialized sequence example
        """
        frames = sequence.tolist()  # convert numpy array to list
        seq_length = sequence.shape[0]  # get the sequence's length

        features_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                         for frame in frames]

        sequence_feats = tf.train.FeatureLists(feature_list=
                                               {"features": tf.train.FeatureList(feature=features_list)})

        seq_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_length]))

        label_bytes_list = self.labels_handler.encode(label)

        label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=label_bytes_list))

        context_features = tf.train.Features(feature={"seq_length": seq_length_feature,
                                                      "label": label_feature})

        ex = tf.train.SequenceExample(context=context_features,
                                      feature_lists=sequence_feats)

        return ex.SerializeToString()

    def write_tf_record(self, writer, feature_vector, label):
        ex = self.encode_sequence_example(feature_vector, label)
        writer.write(ex)

    def write(self, data_dir, record_file_path, logging_freq=50):
        writer = tf.python_io.TFRecordWriter(record_file_path)
        unparsed_examples = self.data_dir_handle(data_dir)

        for n, unparsed_ex in enumerate(unparsed_examples):
            #  get numpy array [seq_length x num_ceps]
            feature_vector = self.features_selector.get_feature_vector(file_path=unparsed_ex['audio_file_path'],
                                                                       feature_settings=self.features_settings)
            #  simply string representation of the label
            label = unparsed_ex['label']
            self.write_tf_record(writer=writer,
                                 feature_vector=feature_vector,
                                 label=label)

            if n % logging_freq == 0:
                tf.logging.info("Processed: %d / %d" % (n, len(unparsed_examples)))

        writer.close()


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

    writer = Writer(config_file=ARGS.writer_config)
    writer.write(data_dir=ARGS.data_dir,
                 record_file_path=ARGS.record_path)
