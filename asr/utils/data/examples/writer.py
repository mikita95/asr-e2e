import argparse
import os
import re
import sys

import tensorflow as tf

import asr.utils.data.examples.features.selector as fsb
import asr.utils.data.examples.labels.handler as lh

FLAGS = None
MODES = ['mfcc', 'fbank', 'raw']


"""
Expected data directory structure:
    data >
        number >
            %format [wav, mp3] >
                [file_name.%format]*
            etc >
                txt.done.data: [( file_name "phrase" )\n]*
Creates directories:
    data >
        number >
            %format >
            etc >
    +       %mode [mfcc, fbank, raw] >
                [file_name.csv]*:
                    header [some metadata]
                    numpy array
"""


class Writer(object):
    def __init__(self, features_selector: fsb.FeatureSelector, labels_handler: lh.LabelsHandler):
        self.features_selector = features_selector
        self.labels_handler = labels_handler

    def _parse_labels_file(self, file_path):
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

    def data_dir_handle(self, data_dir, audio_format='wav'):
        examples = []

        for data_subset_dir_name in os.listdir(data_dir):  # data/001
            subset_format_dir_path = os.path.join(data_dir, data_subset_dir_name, audio_format)  # data/001/wav
            tf.logging.info("Going to dir " + subset_format_dir_path)

            label_file_path = os.path.join(data_dir, data_subset_dir_name, 'etc', 'txt.done.data')
            labels_of_current_subset = self._parse_labels_file(label_file_path)

            for audio_file_name in os.listdir(subset_format_dir_path):
                feature_file_path = os.path.join(subset_format_dir_path, audio_file_name)

                #  append pair (path to the audio file, label text of the file)
                example_name = os.path.splitext(audio_file_name)[0]
                examples.append({'audio_file_path': feature_file_path,
                                 'label': labels_of_current_subset[example_name]})

        return examples

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

    def write(self, data_dir, record_file_path, feature_settings):
        writer = tf.python_io.TFRecordWriter(record_file_path)
        unparsed_examples = self.data_dir_handle(data_dir)
        current = 1
        for unparsed_ex in unparsed_examples:
            #  get numpy array [seq_length x num_ceps]
            feature_vector = \
                self.features_selector.get_feature_vector(file_path=unparsed_ex['audio_file_path'], feature_settings=feature_settings)
            #  simply string representation of the label
            label = unparsed_ex['label']
            self.write_tf_record(writer=writer,
                                 feature_vector=feature_vector,
                                 label=label)
            current += 1
            if current % 50 == 0:
                tf.logging.info("Processed: %d / %d" % (current, len(unparsed_examples)))

        writer.close()


def main(_):
    import asr.models.ctc.labels.handler
    tf.logging.set_verbosity(tf.logging.INFO)

    writer = Writer(features_selector=fsb.get_feature_selector(selector_name=FLAGS.mode),
                    labels_handler=asr.models.ctc.labels.handler.CTCLabelsHandler)

    writer.write(data_dir=FLAGS.data_dir,
                 record_file_path=FLAGS.save_dir,
                 feature_settings={})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='data_process', description='Script to process data')

    parser.add_argument("--data_dir", help="Directory of dataset", type=str)
    parser.add_argument("--save_dir", help="Directory where preprocessed arrays are to be saved", type=str)

    parser.add_argument("--rewrite", help="Rewrite csv output file if exists", type=bool, default=False)

    parser.add_argument("--mode",
                        help="Mode",
                        choices=MODES,
                        type=str,
                        default='mfcc')

    parser.add_argument("--labels_format",
                        help='Labels format',
                        choices=['ctc'],
                        type=str,
                        default='ctc')

    parser.add_argument("--format",
                        help="Format of files",
                        choices=['wav', 'mp3'],
                        type=str,
                        default='wav')

    parser.add_argument("--rate",
                        help="Sample rate of the audio files",
                        type=int,
                        default=16000)

    parser.add_argument("--channels",
                        help="Number of channels of the audio files",
                        type=int,
                        default=1)

    parser.add_argument("--winlen", type=float, default=0.025)
    parser.add_argument("--winstep", type=float, default=0.01)
    parser.add_argument("--numcep", type=int, default=13)
    parser.add_argument("--nfilt", type=int, default=26)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--lowfreq", type=int, default=0)
    parser.add_argument("--highfreq", type=int, default=None)
    parser.add_argument("--ceplifter", type=int, default=22)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--appendEnergy", type=bool, default=True)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)