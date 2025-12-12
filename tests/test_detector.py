import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.detector import ObjectDetector


class TestObjectDetector(unittest.TestCase):
    """Unit tests for ObjectDetector module"""
    
    def setUp(self):
        """Initialize detector before each test"""
        self.detector = ObjectDetector()
        
    def test_initialization(self):
        """Test that detector initializes correctly"""
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.mock_mode)
        
    def test_detect_returns_list(self):
        """Test that detect returns a list"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list)
        
    def test_detect_mock_returns_two_objects(self):
        """Test that mock mode returns 2 objects"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        self.assertEqual(len(detections), 2)
        
    def test_detection_format(self):
        """Test that each detection has required fields"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        
        for det in detections:
            self.assertIn('bbox', det)
            self.assertIn('label', det)
            self.assertIn('score', det)
            self.assertIn('class_id', det)
            
    def test_bbox_format(self):
        """Test that bboxes are in xyxy format"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        
        for det in detections:
            bbox = det['bbox']
            self.assertEqual(len(bbox), 4)
            # x1, y1, x2, y2
            self.assertLess(bbox[0], bbox[2])  # x1 < x2
            self.assertLess(bbox[1], bbox[3])  # y1 < y2
            
    def test_confidence_threshold(self):
        """Test that confidence scores are above threshold"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)
        
        for det in detections:
            self.assertGreaterEqual(det['score'], self.detector.confidence_threshold)


if __name__ == '__main__':
    unittest.main()
