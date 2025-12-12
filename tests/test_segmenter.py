import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.segmentation.segmenter import VideoSegmenter


class TestVideoSegmenter(unittest.TestCase):
    """Unit tests for VideoSegmenter module"""
    
    def setUp(self):
        """Initialize segmenter before each test"""
        self.segmenter = VideoSegmenter()
        
    def test_initialization(self):
        """Test that segmenter initializes correctly"""
        self.assertIsNotNone(self.segmenter)
        self.assertTrue(self.segmenter.mock_mode)
        
    def test_track_objects_returns_list(self):
        """Test that track_objects returns a list"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [
            {'bbox': [100, 100, 200, 200], 'class_id': 0, 'label': 'person'}
        ]
        tracks = self.segmenter.track_objects(0, frame, detections)
        self.assertIsInstance(tracks, list)
        
    def test_track_format(self):
        """Test that each track has required fields"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [
            {'bbox': [100, 100, 200, 200], 'class_id': 0, 'label': 'person'}
        ]
        tracks = self.segmenter.track_objects(0, frame, detections)
        
        for track in tracks:
            self.assertIn('id', track)
            self.assertIn('mask', track)
            self.assertIn('bbox', track)
            self.assertIn('class_id', track)
            
    def test_mask_shape(self):
        """Test that mask has correct shape"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = [
            {'bbox': [100, 100, 200, 200], 'class_id': 0, 'label': 'person'}
        ]
        tracks = self.segmenter.track_objects(0, frame, detections)
        
        for track in tracks:
            mask = track['mask']
            self.assertEqual(mask.shape, (720, 1280))
            self.assertEqual(mask.dtype, np.uint8)
            
    def test_reset(self):
        """Test that reset clears state"""
        self.segmenter.inference_state = {"test": "data"}
        self.segmenter.reset()
        self.assertIsNone(self.segmenter.inference_state)


if __name__ == '__main__':
    unittest.main()
