# coding=utf-8
from unittest import TestCase


class TestCTCUtils(TestCase):
    def test_convert_input_to_ctc_format(self):
        from models.ctc_utils import convert_input_to_ctc_format
        import numpy

        expected_seq_length = 10
        expected_num_cep = 13
        feature_vector = numpy.random.rand(expected_seq_length, expected_num_cep)
        target_text = "привет никита как дела"
        feature_tensor, target, seq_length, original = convert_input_to_ctc_format(feature_vector, target_text)

        self.assertEqual((1, expected_seq_length, expected_num_cep), feature_tensor.shape)
        self.assertEqual([expected_seq_length], seq_length)

    def test_handle_feature_vectors_batch(self):
        from models.ctc_utils import handle_feature_vectors_batch
        import numpy

        min_length = 2
        max_length = 9
        lengths = [i for i in range(min_length, max_length + 1)]
        num_cep = 13

        batch = []

        for l in lengths:
            feature_vector = numpy.random.rand(l, num_cep)  # random vector [current_length x num_cep]
            batch.append([feature_vector, str(l)])

        feature_tensors, sparse_targets, seq_lengths, originals = handle_feature_vectors_batch(batch)

        self.assertEqual((max_length - min_length + 1, max_length, num_cep), feature_tensors.shape)

        self.assertEqual((max_length - min_length + 1, 2), sparse_targets[0].shape)
        self.assertEqual((max_length - min_length + 1,), sparse_targets[1].shape)
        self.assertEqual((2,), sparse_targets[2].shape)
