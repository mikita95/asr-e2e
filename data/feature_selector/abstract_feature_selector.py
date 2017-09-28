from abc import ABC, abstractmethod
from tensorflow.contrib import ffmpeg
from models.ctc.data_input import MFCCSelector, FBANKSelector


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


class FeatureSelector(ABC):

    @abstractmethod
    def _get_feature_vector(self, audio, feature_settings, samples_per_second):
        pass

    def get_feature_vector(self, file_path, feature_settings, file_format='wav',
                           samples_per_second=16000, channel_count=1):
        audio = load_audio_file(file_path, file_format, samples_per_second, channel_count)
        return self._get_feature_vector(audio, feature_settings, samples_per_second)

    @staticmethod
    def get_feature_selector(selector_name='mfcc'):
        if selector_name == 'mfcc':
            return MFCCSelector()
        elif selector_name == 'fbank':
            return FBANKSelector()
        else:
            raise TypeError('No such feature selector.')
