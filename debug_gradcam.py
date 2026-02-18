import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from inference import TireAnalyzer

def test_gradcam():
    print("Initializing Analyzer...")
    analyzer = TireAnalyzer()
    
    img_path = "test_tire_good.jpg"
    if not os.path.exists(img_path):
        # Create dummy image if needed
        print(f"Image {img_path} not found, creating dummy...")
        dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy)

    print(f"Processing image: {img_path}")
    original_img, input_tensor = analyzer.preprocess_image(img_path)
    print(f"Input tensor shape: {input_tensor.shape}")

    print("Generating Grad-CAM...")
    try:
        heatmap = analyzer.get_gradcam(input_tensor, analyzer.wear_model)
        if heatmap is not None:
            print("Success! Heatmap generated.")
            print(f"Heatmap shape: {heatmap.shape}")
        else:
            print("Failure: Heatmap is None.")
    except Exception as e:
        print(f"Exception during Grad-CAM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradcam()
