from python_speech_features import mfcc

from feature_selector.abstract_feature_selector import FeatureSelector


class MFCCSelector(FeatureSelector):
    def _get_feature_vector(self, audio, feature_settings, samples_per_second):
        return mfcc(signal=audio,
                     samplerate=samples_per_second,
                     **feature_settings)
