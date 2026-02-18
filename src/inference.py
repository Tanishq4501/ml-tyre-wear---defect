import tensorflow as tf
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to root then to models
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

WEAR_MODEL_PATH = os.path.join(MODELS_DIR, "tire_wear_model.keras")
VEHICLE_MODEL_PATH = os.path.join(MODELS_DIR, "vehicle_model.keras")

class TireAnalyzer:
    def __init__(self):
        self.wear_model = None
        self.vehicle_model = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(WEAR_MODEL_PATH):
                print(f"Loading Wear Model from {WEAR_MODEL_PATH}...")
                self.wear_model = tf.keras.models.load_model(WEAR_MODEL_PATH)
            else:
                print("Warning: Wear model not found.")

            if os.path.exists(VEHICLE_MODEL_PATH):
                print(f"Loading Vehicle Model from {VEHICLE_MODEL_PATH}...")
                self.vehicle_model = tf.keras.models.load_model(VEHICLE_MODEL_PATH)
            else:
                print("Warning: Vehicle model not found.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def preprocess_image(self, image_path):
        """
        Reads image, resizes to (224, 224), and normalizes.
        Returns: original_image (BGR), processed_tensor (1, 224, 224, 3)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        
        # MobileNetV3 preprocessing
        input_arr = tf.keras.applications.mobilenet_v3.preprocess_input(img_resized.astype(np.float32))
        input_tensor = np.expand_dims(input_arr, axis=0)
        
        return img, input_tensor



    def predict_wear(self, input_tensor):
        if not self.wear_model:
            return "Model Not Loaded", 0.0
        
        preds = self.wear_model.predict(input_tensor)
        cls_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        
        # These class names should match directory names alphabetically
        # defaulting to raw names if not trained yet
        class_names = ['defective', 'good'] 
        result = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        
        return result, confidence

    def predict_vehicle(self, input_tensor):
        if not self.vehicle_model:
            return "Model Not Loaded", 0.0
            
        preds = self.vehicle_model.predict(input_tensor)
        cls_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        
        # Alphabetical order
        class_names = ['Bus', 'Car', 'Truck', 'motorcycle']
        result = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        
        return result, confidence

    def calculate_edge_density(self, image):
        """
        Heuristic: Worn tires are smoother -> Lower edge density.
        New/Good tires have deep treads -> Higher edge density.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Canny Edge Detection
        # Lower thresholds to catch fainter edges of worn treads
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate ratio of edge pixels to total pixels
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return edge_density



    def estimate_tread_depth(self, image):
        """
        Estimates tread depth using groove analysis.
        Strategy: Use adaptive thresholding to find dark grooves.
                  Calculate the average width and density of these grooves.
                  Map density to a heuristic mm value (calibration needed for real precision).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Threshold to isolate grooves
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Feature 1: Groove Density
        groove_density = np.sum(opening > 0) / opening.size
        
        # Heuristic Mapping to MM (Requires calibration with real data)
        # Assumption: Max density ~0.3 -> 8mm, Min density ~0.05 -> 1.5mm
        estimated_depth_mm = 1.5 + (groove_density / 0.3) * (8.0 - 1.5)
        estimated_depth_mm = min(max(estimated_depth_mm, 1.5), 9.0) # Clamp
        
        return estimated_depth_mm, groove_density

    def calculate_tread_coverage(self, image):
        """
        Calculates percentage of tread remaining vs bald surface.
        """
        # Similar to edge density but focuses on block segmentation
        return self.calculate_edge_density(image) * 100 * 2.5 # Scale factor heuristic

    def get_gradcam(self, input_tensor, model, layer_name=None):
        """
        Generates Standard Grad-CAM heatmap.
        """
        if not model:
            return None

        # Robust Layer Finding for MobileNetV3/ResNet
        if layer_name is None:
            # Try specific known layers for common models first
            possible_layers = ['Conv_1', 'top_conv', 'conv5_block3_out', 'expanded_conv_6/expand']
            for name in possible_layers:
                try:
                    if model.get_layer(name):
                        layer_name = name
                        break
                except ValueError:
                    continue
            
            # Fallback: Find last 4D layer
            if layer_name is None:
                for layer in reversed(model.layers):
                    if len(layer.output.shape) == 4:
                        layer_name = layer.name
                        break
        
        if layer_name is None:
            print("Grad-CAM: Could not identify a 4D output layer.")
            return None
            
        print(f"Grad-CAM: Using layer '{layer_name}'")

        # Standard Grad-CAM Model
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            cls_idx = np.argmax(predictions[0])
            loss = predictions[:, cls_idx]

        # Gradients of class score w.r.t. conv outputs
        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]

        # Global Average Pooling of Gradients (Standard Grad-CAM)
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Weighted sum of feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        # Resize and Check
        cam = cam.numpy()
        if cam is None or np.max(cam) == 0:
            return None

        cam = cv2.resize(cam, IMG_SIZE)
        cam = np.maximum(cam, 0) # ReLU
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return heatmap

if __name__ == "__main__":
    # Test
    analyzer = TireAnalyzer()
    print("Analyzer initialized.")
