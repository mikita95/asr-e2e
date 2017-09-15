import argparse

import tensorflow as tf
from tensorflow.contrib import ffmpeg

FLAGS = None


def load_file(file_path, file_format,
              samples_per_second, channel_count,
              samples_per_second_tensor=None, feed_dict=None):
    """
    Loads an audio file and decodes it.
    Args:
        file_path: The path of the input file.
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
    with tf.Session():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='data_process', description='Script to process data')
    parser.add_argument("data_dir", help="Directory of dataset", type=str)
    parser.add_argument("save_dir", help="Directory where preprocessed arrays are to be saved", type=str)
    parser.add_argument("-m", "--mode", help="Mode",
                         choices=['mfcc', 'fbank', 'raw'],
                         type=str, default='mfcc')