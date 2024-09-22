import os

import pandas as pd
import unittest
from tempfile import NamedTemporaryFile

from .. import score, vectorize_sentences


class Test(unittest.TestCase):

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    test_assets_dir = curr_dir + '/assets/'

    def setUp(self) -> None:
        # If sample.pfl does not exists
        file_path = self.test_assets_dir + 'sample.pkl'
        if not os.path.exists(file_path):
            self.create_sample_pkl()

    def create_sample_pkl(self, csv_file='sample.csv', pkl_file='sample.pkl'):
        """ Create a sample pkl file used for the tests """

        df = vectorize_sentences.read_file(self.test_assets_dir + csv_file)
        df = vectorize_sentences.vectorize_text(df)
        vectorize_sentences.pickle_data(
            df, self.test_assets_dir + pkl_file)

    def test_check_file_exists(self):
        # Check if the file exists
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = curr_dir + '/assets/sample.csv'
        self.assertTrue(score.check_file_exists(file_path))

        # Test exception
        with self.assertRaises(FileNotFoundError):
            score.check_file_exists('/tmp/missing.csv')

    def test_load_data(self):
        # Load the DataFrame
        file_path = self.curr_dir + '/assets/sample.pkl'
        df = score.load_data(file_path)

        # Check the type of the returned object
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_cosine_similarity(self):
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        self.assertEqual(score.cosine_similarity(v1, v2), 1.0)

    def test_jaccard_similarity(self):
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        self.assertEqual(score.jaccard_similarity(v1, v2), 1.0)

    def test_manhattan_similarity(self):
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        self.assertEqual(score.manhattan_similarity(v1, v2), 1.0)

    def test_compare_vectors(self):
        # Unpickle the DataFrame
        file_path = self.test_assets_dir + 'sample.pkl'
        df = score.load_data(file_path)

        # Compare vectors
        vectors = df['vectors'][0]

        # Test cosine similarity
        similarities = score.compare_vectors(vectors, df, metric='cosine')
        self.assertEqual(len(similarities), 3)

        # Test jaccard similarity
        similarities = score.compare_vectors(vectors, df, metric='jaccard')
        self.assertEqual(len(similarities), 3)

        # Test manhattan similarity
        similarities = score.compare_vectors(vectors, df, metric='manhattan')
        self.assertEqual(len(similarities), 3)

    def test_display_results(self):
        df = pd.DataFrame([{'Text': 'This is a test sentence'}])

        res = score.display_results(df)
        self.assertIsInstance(res, pd.DataFrame)

    def test_plot_results(self):
        # Unpickle the DataFrame
        file_path = self.test_assets_dir + 'sample.pkl'
        df = score.load_data(file_path)

        df = score.compare_vectors(df['vectors'][0], df)

        image_path = NamedTemporaryFile().name + '.png'
        score.plot_results(df, 3, image_path)

        self.assertTrue(os.path.exists(image_path))
