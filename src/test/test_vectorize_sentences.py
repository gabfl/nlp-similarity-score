import os

import numpy as np
import pandas as pd
import unittest
from tempfile import NamedTemporaryFile

from .. import vectorize_sentences


class Test(unittest.TestCase):

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    def test_read_file(self):
        # Read the file
        file_path = self.curr_dir + '/assets/sample.csv'
        df = vectorize_sentences.read_file(file_path)

        # Check the type of the returned object
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_vectorize(self):
        text = "This is a test sentence"
        self.assertIsInstance(vectorize_sentences.vectorize(text), np.ndarray)
        self.assertEqual(len(vectorize_sentences.vectorize(text)), 768)
        self.assertTrue(vectorize_sentences.vectorize(text).any())

    def test_vectorize_text(self):
        df = pd.DataFrame([{'Text': 'This is a test sentence'}])
        df = vectorize_sentences.vectorize_text(df)
        self.assertEqual(len(df['vectors'][0]), 768)

    def test_pickle_data(self):
        # Create temporary file
        file_path = NamedTemporaryFile().name
        df = pd.DataFrame([{'Text': 'This is a test sentence'}])
        vectorize_sentences.pickle_data(df, file_path)

        # Check if the file exists
        self.assertTrue(os.path.exists(file_path))
