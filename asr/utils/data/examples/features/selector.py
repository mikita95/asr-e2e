from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

import tensorflow as tf
from python_speech_features import fbank
from python_speech_features import mfcc


class FeatureSelector(ABC):
    @abstractmethod
    def get_features_vector(self, raw_data, settings=None):
        pass


class AudioFeatureSelector(FeatureSelector):
    @abstractmethod
    def get_features_vector(self, raw_audio_data, settings=None) -> np.ndarray:
        """Computes features vector for raw audio data

        Args:
            raw_audio_data (numpy.ndarray): A numpy array corresponding to audio file [L x C]
            settings (dict(str -> obj)): Dictionary for settings of features

        Returns:
            features_vector (numpy.ndarray): A numpy array of features

        """
        pass

    @staticmethod
    def _load_audio_file(audio_file_path):
        """Loads an audio file and decodes it.

        Args:
            audio_file_path (str): The path of the input file.

        Returns:
            audio_result (numpy.ndarray): A numpy array corresponding to audio file [L x C]
            sample_rate (int): Sample rate of the read audio file

        """
        import soundfile as sf
        tf.logging.debug("Handling audio file: " + audio_file_path)
        audio_result, sample_rate = sf.read(file=audio_file_path)
        return audio_result, sample_rate

    def get_audio_features_vector(self, file_path, settings) -> (tf.train.FeatureLists, int):
        audio, _ = self._load_audio_file(audio_file_path=file_path)
        features_vector = self.get_features_vector(raw_audio_data=audio, settings=settings)

        frames = features_vector.tolist()  # convert numpy array to list
        seq_length = features_vector.shape[0]  # get the sequence's length

        # Make sequence features
        features_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame)) for frame in frames]
        sequence_feats = tf.train.FeatureLists(feature_list={"features": tf.train.FeatureList(feature=features_list)})
        return sequence_feats, seq_length

    @abstractmethod
    def get_sequence_feature(self):
        return  {
            "features": tf.FixedLenSequenceFeature([13, ], dtype=tf.float32)
        }


class FBANKAudioSelector(AudioFeatureSelector):
    def get_features_vector(self, raw_audio_data, settings=None):
        return fbank(signal=raw_audio_data,
                     samplerate=int(settings['samplerate']),
                     winlen=float(settings['winlen']),
                     winstep=float(settings['winstep']),
                     nfilt=int(settings['nfilt']),
                     nfft=int(settings['nfft']),
                     lowfreq=int(settings['lowfreq']),
                     highfreq=int(settings['highfreq']),
                     preemph=float(settings['preemph']))


class MFCCAudioSelector(AudioFeatureSelector):
    def get_features_vector(self, raw_audio_data, settings=None):
        import numpy as np
        audio_data = raw_audio_data - np.mean(raw_audio_data)
        audio_data /= np.max(audio_data)
        return mfcc(signal=audio_data,
                    samplerate=int(settings['samplerate']),
                    winlen=float(settings['winlen']),
                    winstep=float(settings['winstep']),
                    numcep=int(settings['numcep']),
                    nfilt=int(settings['nfilt']),
                    nfft=int(settings['nfft']),
                    lowfreq=int(settings['lowfreq']),
                    highfreq=int(settings['highfreq']),
                    ceplifter=int(settings['ceplifter']),
                    preemph=float(settings['preemph']),
                    appendEnergy=bool(settings['append_energy']))


class AudioSelector(Enum):
    MFCC = 'mfcc'
    FBANK = 'fbank'


def get_audio_feature_selector(audio_selector_name='mfcc'):
    if audio_selector_name == AudioSelector.MFCC.value:
        return MFCCAudioSelector()
    elif audio_selector_name == AudioSelector.FBANK.value:
        return FBANKAudioSelector()
    else:
        raise TypeError('No such feature selector.')
