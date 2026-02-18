import sys
import os
import unittest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import TireAnalyzer

class TestModelLoad(unittest.TestCase):
    def test_analyzer_initialization(self):
        """
        Test if TireAnalyzer initializes without crashing.
        Note: Models might not be present yet, but it should handle that gracefully.
        """
        analyzer = TireAnalyzer()
        self.assertIsInstance(analyzer, TireAnalyzer)
        
    def test_preprocessing(self):
        """
        Test if preprocessing returns correct shape.
        """
        analyzer = TireAnalyzer()
        # Create a dummy image
        dummy_img_path = "temp_test_img.jpg"
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(dummy_img_path, dummy_img)
        
        try:
            original, tensor = analyzer.preprocess_image(dummy_img_path)
            self.assertEqual(tensor.shape, (1, 224, 224, 3))
            self.assertEqual(original.shape, (300, 300, 3))
        finally:
            if os.path.exists(dummy_img_path):
                os.remove(dummy_img_path)

if __name__ == '__main__':
    unittest.main()
