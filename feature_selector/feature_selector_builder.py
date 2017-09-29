from feature_selector.fbank_selector import FBANKSelector
from feature_selector.mfcc_selector import MFCCSelector


def get_feature_selector(selector_name='mfcc'):
    if selector_name == 'mfcc':
        return MFCCSelector()
    elif selector_name == 'fbank':
        return FBANKSelector()
    else:
        raise TypeError('No such feature selector.')