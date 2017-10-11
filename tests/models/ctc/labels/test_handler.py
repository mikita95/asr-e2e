from unittest import TestCase
import pkg_resources


class TestCTCLabelsHandler(TestCase):
    def test_handle(self):
        import asr.models.ctc.labels.handler as h
        #  print(pkg_resources.resource_filename('tests.resources', 'russian.ini'))
        handler = h.CTCLabelsHandler(alphabet_file='russian.ini')
        print(handler.alphabet)
        print(handler.handle("привет как дела"))
