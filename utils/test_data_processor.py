# coding=utf-8
from unittest import TestCase
from contextlib import contextmanager
import os


@contextmanager
def mock_file(file_path, content=''):
    with open(file_path, 'w', encoding='utf8') as f:
        f.write(content)
    yield file_path
    try:
        os.remove(file_path)
    except Exception:
        pass


class TestDataProcessor(TestCase):
    def test_parse_labels_file(self):
        from util.data_processor import parse_labels_file

        content = '( essv_001 "зима была долгой и вьюжной" )\n( essv_002 "это сейчас пройдет" )'

        expected = {'essv_001': "зима была долгой и вьюжной", 'essv_002': "это сейчас пройдет"}
        file_path = 'txt.done.data'

        with mock_file(file_path=file_path, content=content):
            actual = parse_labels_file(file_path)
            self.assertDictEqual(expected, actual)

    def test_pad_feature_vectors(self):
        from util.data_processor import pad_feature_vectors
        import numpy

        # test data:
        min_length = 2
        max_length = 12
        lengths = [i for i in range(min_length, max_length + 1)]
        num_cep = 13

        examples = []

        for l in lengths:
            feature_vector = numpy.random.rand(l, num_cep)  # random vector [num_cep x current_length]
            examples.append([feature_vector, str(l)])

        padded_examples = pad_feature_vectors(examples)
        for i in range(0, len(examples)):
            self.assertEqual((lengths[-1], num_cep), padded_examples[i][0].shape)
            self.assertEqual(str(lengths[i]), padded_examples[i][1])
