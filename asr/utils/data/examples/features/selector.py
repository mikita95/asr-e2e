from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf
from python_speech_features import fbank
from python_speech_features import mfcc


class Selector(Enum):
    MFCC = 'mfcc'
    FBANK = 'fbank'


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
    def _get_feature_vector(self, audio, feature_settings):
        pass

    def get_feature_vector(self, file_path, feature_settings, file_format='wav'):
        audio = load_audio_file(file_path)
        return self._get_feature_vector(audio, feature_settings)


class FBANKSelector(FeatureSelector):
    def _get_feature_vector(self, audio, feature_settings):
        return fbank(signal=audio,
                     samplerate=int(feature_settings['samplerate']),
                     winlen=float(feature_settings['winlen']),
                     winstep=float(feature_settings['winstep']),
                     nfilt=int(feature_settings['nfilt']),
                     nfft=int(feature_settings['nfft']),
                     lowfreq=int(feature_settings['lowfreq']),
                     highfreq=int(feature_settings['highfreq']),
                     preemph=float(feature_settings['preemph']))


class MFCCSelector(FeatureSelector):
    def _get_feature_vector(self, audio, feature_settings):
        import numpy as np
        audio_data = audio - np.mean(audio)
        audio_data /= np.max(audio_data)
        return mfcc(signal=audio_data,
                    samplerate=int(feature_settings['samplerate']),
                    winlen=float(feature_settings['winlen']),
                    winstep=float(feature_settings['winstep']),
                    numcep=int(feature_settings['numcep']),
                    nfilt=int(feature_settings['nfilt']),
                    nfft=int(feature_settings['nfft']),
                    lowfreq=int(feature_settings['lowfreq']),
                    highfreq=int(feature_settings['highfreq']),
                    ceplifter=int(feature_settings['ceplifter']),
                    preemph=float(feature_settings['preemph']),
                    appendEnergy=bool(feature_settings['append_energy']))


def get_feature_selector(selector_name='mfcc'):
    if selector_name == Selector.MFCC.value:
        return MFCCSelector()
    elif selector_name == Selector.FBANK.value:
        return FBANKSelector()
    else:
        raise TypeError('No such feature selector.')
