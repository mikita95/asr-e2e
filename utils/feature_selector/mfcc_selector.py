from python_speech_features import mfcc

from utils.feature_selector.abstract_feature_selector import FeatureSelector


class MFCCSelector(FeatureSelector):

    def _get_feature_vector(self, audio, feature_settings, samples_per_second):
        import numpy as np
        audio_data = audio - np.mean(audio)
        audio_data /= np.max(audio_data)
        return mfcc(signal=audio_data,
                    samplerate=samples_per_second,
                    **feature_settings)
