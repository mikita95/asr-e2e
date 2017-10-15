from python_speech_features import fbank

from asr.utils.data.examples.features.selector import FeatureSelector


class FBANKSelector(FeatureSelector):
    def _get_feature_vector(self, audio, feature_settings, samples_per_second):
        return fbank(signal=audio,
                     samplerate=samples_per_second,
                     **feature_settings)
