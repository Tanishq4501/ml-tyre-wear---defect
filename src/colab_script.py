# ==========================================
# 🚀 Google Colab Training Script for YOLOv8
# ==========================================
# Instructions:
# 1. Zip your local dataset folder: 'F:\TechS\ML_Tire\data\raw\defect.v2i.yolov8' -> 'defect_dataset.zip'
# 2. Open Google Colab: https://colab.research.google.com/
# 3. Upload 'defect_dataset.zip' to the Colab files area (left sidebar).
# 4. Copy-paste this entire script into a code cell and run it.
# 5. After training, the script will automatically download 'runs.zip' containing your trained model.

import os
import shutil
import yaml
from ultralytics import YOLO

# --- Step 1: Install Dependencies ---
print("Installing ultralytics...")
os.system('pip install ultralytics')

# --- Step 2: Unzip Dataset ---
dataset_zip = "defect_dataset.zip" # Ensure this matches your uploaded file name
dataset_dir = "/content/dataset"

if os.path.exists(dataset_zip):
    print(f"Unzipping {dataset_zip}...")
    shutil.unpack_archive(dataset_zip, dataset_dir)
else:
    raise FileNotFoundError(f"Please upload '{dataset_zip}' to Colab files!")

# --- Step 3: Find data.yaml (Handle nested folders) ---
import glob
print("Searching for data.yaml...")
yaml_files = glob.glob(f"{dataset_dir}/**/data.yaml", recursive=True)

if not yaml_files:
    print(f"ERROR: Could not find data.yaml in {dataset_dir}")
    print("Directory structure:")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        for f in files:
            print(f'{indent}    {f}')
    raise FileNotFoundError("data.yaml not found! Check the structure above.")

yaml_path = yaml_files[0]
dataset_root = os.path.dirname(yaml_path)
print(f"Found config at: {yaml_path} (Root: {dataset_root})")

# --- Step 4: Fix data.yaml Paths ---
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Update paths to point to where we found data.yaml
# We assume standard YOLO structure relative to data.yaml location
data['path'] = dataset_root # Optional but helpful
data['train'] = f"{dataset_root}/train/images"
# Use test set for validation if valid is missing
val_path = f"{dataset_root}/valid/images"
if not os.path.exists(val_path):
     val_path = f"{dataset_root}/test/images"

data['val'] = val_path
data['test'] = f"{dataset_root}/test/images"

# Save updated yaml
with open(yaml_path, 'w') as f:
    yaml.dump(data, f)
print("Updated data.yaml paths for Colab.")

# --- Step 5: Train Model ---
print("Starting Training...")
# Load a pretrained YOLOv8n-seg model
model = YOLO('yolov8n-seg.pt') 

# Train
results = model.train(
    data=yaml_path,
    epochs=50,       # Adjust epochs as needed
    imgsz=640,
    batch=16,
    name='tire_defect_segmentation'
)

# --- Step 6: Zip and Download Results ---
print("Zipping results...")
shutil.make_archive("/content/runs", 'zip', "/content/runs")

from google.colab import files
files.download("/content/runs.zip")

print("Download started! Upload 'runs.zip' content back to your local 'runs/' folder.")
