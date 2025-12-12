import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering.identifier import VisualIdentifier


class TestVisualIdentifier(unittest.TestCase):
    """Unit tests for VisualIdentifier module"""
    
    def setUp(self):
        """Initialize identifier before each test"""
        self.identifier = VisualIdentifier()
        
    def test_initialization(self):
        """Test that identifier initializes correctly"""
        self.assertIsNotNone(self.identifier)
        self.assertTrue(self.identifier.mock_mode)
        
    def test_extract_embeddings_returns_array(self):
        """Test that extract_embeddings returns numpy array"""
        crops = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        embeddings = self.identifier.extract_embeddings(crops)
        self.assertIsInstance(embeddings, np.ndarray)
        
    def test_embeddings_shape(self):
        """Test that embeddings have correct shape"""
        crops = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        embeddings = self.identifier.extract_embeddings(crops)
        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.shape[1], 768)  # SigLIP base dimension
        
    def test_cluster_embeddings_returns_labels(self):
        """Test that clustering returns labels"""
        embeddings = np.random.rand(10, 768).astype(np.float32)
        labels = self.identifier.cluster_embeddings(embeddings, n_clusters=2)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), 10)
        
    def test_cluster_labels_valid(self):
        """Test that cluster labels are in valid range"""
        embeddings = np.random.rand(10, 768).astype(np.float32)
        labels = self.identifier.cluster_embeddings(embeddings, n_clusters=2)
        
        unique_labels = set(labels)
        self.assertTrue(unique_labels.issubset({0, 1}))
        
    def test_empty_crops(self):
        """Test handling of empty crop list"""
        embeddings = self.identifier.extract_embeddings([])
        self.assertEqual(len(embeddings), 0)


if __name__ == '__main__':
    unittest.main()
