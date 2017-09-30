# coding=utf-8
import argparse

import tensorflow as tf
from tensorflow.contrib import ffmpeg
from python_speech_features import mfcc
from python_speech_features import fbank
from python_speech_features import delta
from python_speech_features import logfbank
import sys
import os
import numpy
import re
from random import shuffle

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


def load_batched_data(data_path, batch_size, mode='mfcc', randomize=False):
    """
    Generator for batches of data examples
    Args:
        data_path: path to the root of the data
        batch_size: size of the desired batches
        mode: format of the feature vectors, look MODE list
        randomize: is need to be shuffled

    Returns: iterator to the batches,
            where one batch is a list of pairs <path_to_feature_file, target_text> of batch_size size
            the last batch can be non-full

    """
    all_examples = []
    for examples_subset_dir_name in os.listdir(data_path):  # data/001
        mode_dir_path = os.path.join(data_path, examples_subset_dir_name, mode)  # data/001/mfcc

        labels_file_path = os.path.join(data_path, examples_subset_dir_name, "etc", "txt.done.data")  # data/001/etc/txt.done.data
        labels = parse_labels_file(labels_file_path)

        for example_file_name in os.listdir(mode_dir_path):  # essv_001.csv
            example_name = os.path.splitext(example_file_name)[0]  # essv_001

            example_file_path = os.path.join(mode_dir_path, example_file_name)  # data/001/mfcc/essv_001.csv
            if example_name == '.csv':
                print(example_file_path)

            all_examples.append([example_file_path, labels[example_name]])

    if randomize:
        shuffle(all_examples)

    for i in range(0, len(all_examples), batch_size):
        yield all_examples[i:i + batch_size]


def parse_example_file(example_file_path):
    """
    Parses file with feature data
    Args:
        example_file_path: path to the file
    Returns: numpy array
        E.g: in mfcc mode output is [seq_length x numcep] array
    """
    return numpy.loadtxt(fname=example_file_path, delimiter=",")


def pad_feature_vectors(examples):
    """
    Adds padding to the sequences, makes the lengths of them equal
    Args:
        examples: list E of pairs of <feature_vector, target_text>
                  where feature_vector is [seq_length x num_cep] in E

    Returns: list of pairs of <padded_feature_vector, target_text>
             where padded_feature_vector is [max(seq_length) x num_cep]

    """
    lengths = numpy.asarray([f[0].shape[0] for f in examples], dtype=numpy.int64)
    max_len = numpy.max(lengths)

    result = []
    for e in examples:
        padded_feature_vector = numpy.zeros(shape=[max_len, e[0].shape[1]])
        padded_feature_vector[:e[0].shape[0], :e[0].shape[1]] = e[0]
        result.append([padded_feature_vector, e[1]])
    return result


def load_audio_file(file_path, file_format,
                    samples_per_second, channel_count,
                    samples_per_second_tensor=None, feed_dict=None):
    """
    Loads an audio file and decodes it.
    Args:
        file_path:   The path of the input file.
        file_format: The desired sample rate in the output tensor.
        samples_per_second: The sample rate of the audio file
        channel_count: The desired channel count in the output tensor.
        samples_per_second_tensor: The value to pass to the corresponding parameter in the instantiated
           `decode_audio` op. If not provided, will default to a constant value of `samples_per_second`.
           Useful for providing a placeholder.
        feed_dict: Used when evaluating the `decode_audio` op. If not provided, will be empty.
           Useful when providing a placeholder for `samples_per_second_tensor`.

    Returns: A numpy array corresponding to audio file.
    """

    if samples_per_second_tensor is None:
        samples_per_second_tensor = samples_per_second

    with open(file_path, 'rb') as f:
        contents = f.read()

    audio_op = ffmpeg.decode_audio(
        contents,
        file_format=file_format,
        samples_per_second=samples_per_second_tensor,
        channel_count=channel_count)

    audio_result = audio_op.eval(feed_dict=feed_dict or {})

    assert len(audio_result.shape) == 2, \
        'Expected audio shape length equals 2 but found %d(found_shape)' % len(audio_result.shape)
    assert audio_result.shape[1] == channel_count, \
        'Expected channel count %d(exp_ch) but found %d(found_ch)' % (channel_count, audio_result.shape[1])

    return audio_result


def get_feature(file_path, feature_settings, file_format='wav',
                samples_per_second=16000, channel_count=1, mode='mfcc'):
    audio = load_audio_file(file_path, file_format, samples_per_second, channel_count)
    if mode == 'mfcc':
        return mfcc(signal=audio,
                    samplerate=samples_per_second,
                    winlen=feature_settings["winlen"],
                    winstep=feature_settings["winstep"],
                    numcep=feature_settings["numcep"],
                    nfilt=feature_settings["nfilt"],
                    nfft=feature_settings["nfft"],
                    lowfreq=feature_settings["lowfreq"],
                    highfreq=feature_settings["highfreq"],
                    ceplifter=feature_settings["ceplifter"],
                    preemph=feature_settings["preemph"],
                    appendEnergy=feature_settings["appendEnergy"])
    if mode == 'fbank':
        return fbank(signal=audio,
                     samplerate=samples_per_second,
                     winlen=feature_settings["winlen"],
                     winstep=feature_settings["winstep"],
                     nfilt=feature_settings["nfilt"],
                     nfft=feature_settings["nfft"],
                     lowfreq=feature_settings["lowfreq"],
                     highfreq=feature_settings["highfreq"],
                     preemph=feature_settings["preemph"])
    if mode == 'row':
        return audio
    return None


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.InteractiveSession()
    for file_dir in os.listdir(FLAGS.data_dir):
        tf.logging.info("Going to dir " + file_dir)

        file_format_path = os.path.join(FLAGS.data_dir, file_dir, FLAGS.format)
        for file_path in os.listdir(file_format_path):
            tf.logging.info("Start converting file " + file_path)

            output_name = os.path.splitext(file_path)[0] + ".csv"
            output_dir = os.path.join(FLAGS.save_dir, file_dir, FLAGS.mode)
            output_path = os.path.join(output_dir, output_name)

            if os.path.exists(output_path) and not FLAGS.rewrite:
                tf.logging.info("File " + output_path + " has been already created")
                continue

            features = get_feature(file_path=os.path.join(file_format_path, file_path),
                                   feature_settings=vars(FLAGS),
                                   file_format=FLAGS.format,
                                   samples_per_second=FLAGS.rate,
                                   channel_count=FLAGS.channels,
                                   mode=FLAGS.mode)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            tf.logging.info("Saving into " + output_path)
            numpy.savetxt(output_path, features, delimiter=",", header=str(vars(FLAGS)))


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
