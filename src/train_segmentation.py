from ultralytics import YOLO
import os
import yaml

# Configuration
DATASET_DIR = r"F:\TechS\ML_Tire\data\raw\defect.v2i.yolov8"
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
EPOCHS = 50
IMGSZ = 640

def fix_yaml_paths():
    """Ensures data.yaml has absolute paths to avoid YOLO errors."""
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: {DATA_YAML_PATH} not found.")
        return False

    with open(DATA_YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)

    # Convert to absolute paths
    # If 'valid' folder is missing, use 'test' for validation
    train_path = os.path.join(DATASET_DIR, 'train', 'images')
    test_path = os.path.join(DATASET_DIR, 'test', 'images')
    valid_path = os.path.join(DATASET_DIR, 'valid', 'images')

    if not os.path.exists(valid_path) and os.path.exists(test_path):
        print("Warning: 'valid' folder not found. Using 'test' folder for validation.")
        valid_path = test_path

    data['train'] = train_path
    data['val']   = valid_path
    data['test']  = test_path

    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Updated {DATA_YAML_PATH} with absolute paths.")
    return True

def train_yolo():
    # Fix paths first
    if not fix_yaml_paths():
        return

    # Load a Segmentation model
    print("Loading YOLOv8n-seg (Segmentation)...")
    model = YOLO('yolov8n-seg.pt')  

    print(f"Starting training on {DATA_YAML_PATH}...")
    try:
        results = model.train(data=DATA_YAML_PATH, epochs=EPOCHS, imgsz=IMGSZ)
        print("Training complete!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    train_yolo()
