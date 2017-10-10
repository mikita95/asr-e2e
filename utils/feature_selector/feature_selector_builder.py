from utils.feature_selector.fbank_selector import FBANKSelector

from utils.feature_selector.mfcc_selector import MFCCSelector

from enum import Enum, auto


class Selector(Enum):
    MFCC = 'mfcc'
    FBANK = 'fbank'


def get_feature_selector(selector_name='mfcc'):
    if selector_name == Selector.MFCC:
        return MFCCSelector()
    elif selector_name == Selector.FBANK:
        return FBANKSelector()
    else:
        raise TypeError('No such feature selector.')