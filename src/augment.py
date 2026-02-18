import os
import glob
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import shutil

# Configuration
RAW_DATA_ROOT = r"F:\TechS\ML_Tire\data\raw"
PROCESSED_DATA_DIR = r"F:\TechS\ML_Tire\data\processed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUGMENTATION_FACTOR = 5

# Define source mappings: "TargetCategory" -> ["SourcePath1", "SourcePath2", ...]
SOURCE_DIRS = {
    "tire_wear": [
        os.path.join(RAW_DATA_ROOT, "tyre_quality_classification", "Digital images of defective and good condition tyres"),
        os.path.join(RAW_DATA_ROOT, "Tyre dataset"),
        # Add texture recognition if relevant, but maybe structure differs
        # os.path.join(RAW_DATA_ROOT, "Tire_Texture_Recognition"), 
    ],
    "vehicle_type": [
        os.path.join(RAW_DATA_ROOT, "Vehicle_Type_Recognition", "Dataset")
    ]
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def augment_image(image, save_dir, prefix):
    """
    Applies augmentation to a single image and saves the results.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    img_array = img_to_array(image)
    img_array = img_array.reshape((1,) + img_array.shape)

    i = 0
    # Use 'jpeg' format to avoid some PIL errors with other formats
    for batch in datagen.flow(img_array, batch_size=1, 
                              save_to_dir=save_dir, 
                              save_prefix=prefix, 
                              save_format='jpeg'):
        i += 1
        if i >= AUGMENTATION_FACTOR:
            break

def process_category(category_name, source_paths):
    """
    Process images for a specific category (e.g., 'tire_wear') from multiple sources.
    """
    print(f"\nProcessing Category: {category_name}")
    
    dest_category_root = os.path.join(PROCESSED_DATA_DIR, category_name)
    ensure_dir(dest_category_root)

    for source_path in source_paths:
        print(f"Source: {source_path}")
        
        if not os.path.exists(source_path):
            print(f"ERROR: Source path not found: {source_path}")
            continue

        # Get all subdirectories (classes) in the source path
        classes = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
        
        if not classes:
            print(f"WARNING: No class subdirectories found in {source_path}")
            continue

        for class_name in classes:
            class_raw_dir = os.path.join(source_path, class_name)
            # Normalize class name for destination (lowercase)
            # Handle potential mapping (e.g. 'Defective' -> 'defective')
            # If classes match roughly, just lower()
            normalized_class_name = class_name.lower()
            
            # Additional mapping if needed
            if normalized_class_name == "normal": normalized_class_name = "good"
            if normalized_class_name == "bald": normalized_class_name = "defective" # Example
            
            class_processed_dir = os.path.join(dest_category_root, normalized_class_name)
            ensure_dir(class_processed_dir)
            
            print(f"  -> Class: {class_name} -> {class_processed_dir}")
            
            # Grab common image formats
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(class_raw_dir, ext)))
                image_files.extend(glob.glob(os.path.join(class_raw_dir, ext.upper())))
                
            print(f"     Found {len(image_files)} images.")
            
            count = 0
            for img_file in image_files:
                try:
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    # Check if already processed to avoid re-doing work if safe
                    # But for now, let's just overwrite to be safe with updates
                    
                    # Load and Resize
                    img = load_img(img_file, target_size=IMG_SIZE)
                    
                    # Save original resized
                    # Add simple prefix to avoid collision from different datasets
                    # e.g. using hash or parent folder name
                    prefix = os.path.basename(source_path)[:4] 
                    save_name = f"orig_{prefix}_{base_name}.jpg"
                    
                    img.save(os.path.join(class_processed_dir, save_name))
                    
                    # Augment
                    augment_image(img, class_processed_dir, f"aug_{prefix}_{base_name}")
                    count += 1
                except Exception as e:
                    print(f"     Error processing {os.path.basename(img_file)}: {e}")
                    
            print(f"     Processed {count} images successfully.")

if __name__ == "__main__":
    print("--- Starting Data Augmentation Pipeline ---")
    
    for category, paths in SOURCE_DIRS.items():
        process_category(category, paths)
    
    print("\n--- Pipeline Complete ---")
