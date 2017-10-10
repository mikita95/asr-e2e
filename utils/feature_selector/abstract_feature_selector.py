from abc import ABC, abstractmethod
import tensorflow as tf


def load_audio_file(file_path):
    import soundfile as sf
    """
    Loads an audio file and decodes it.
    Args:
        file_path:   The path of the input file.
    Returns: A numpy array corresponding to audio file.
    """
    tf.logging.info("Handling audio file: " + file_path)

    audio_result, sample_rate = sf.read(file=file_path)

    return audio_result


class FeatureSelector(ABC):

    @abstractmethod
    def _get_feature_vector(self, audio, feature_settings, samples_per_second):
        pass

    def get_feature_vector(self, file_path, feature_settings, file_format='wav',
                           samples_per_second=16000, channel_count=1):
        audio = load_audio_file(file_path)
        return self._get_feature_vector(audio, feature_settings, samples_per_second)

