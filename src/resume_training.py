from ultralytics import YOLO
import os

# Path to the last checkpoint (adjust if needed)
checkpoint_path = r"F:\TechS\ML_Tire\runs\segment\train5\weights\last.pt"

if os.path.exists(checkpoint_path):
    print(f"Resuming training from {checkpoint_path}...")
    model = YOLO(checkpoint_path)
    
    # Resume training
    # Note: 'resume=True' automatically loads training state (epochs, optimizer, etc.)
    results = model.train(resume=True)
else:
    print(f"Checkpoint not found at {checkpoint_path}")
