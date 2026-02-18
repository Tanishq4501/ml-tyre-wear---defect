# Tire Damage & Quality Inspection POC

## Overview
This AI-powered application predicts tire wear levels (Good vs Defective) and identifies vehicle types from uploaded images. It uses valid Deep Learning models (MobileNetV3 for Wear, ResNet50V2 for Vehicle Type) and provides explainability via Grad-CAM and texture analysis.

## Setup
1.  Ensure you have the dependencies installed:
    ```bash
    pip install -r requirements.txt
    ```
2.  Data Preparation (if not already done):
    - Extract images to `data/raw/`
    - Run `python src/augment.py`

## Running the App
Double-click `run_app.bat` or run:
```bash
streamlit run app/app.py
```

## Features
- **Tire Wear Prediction**: Classifies tires as "Good" or "Defective".
- **Vehicle Classification**: Identifies if the vehicle is a Car, Truck, Bus, or Motorcycle.
- **Explainability**:
    - **Grad-CAM**: Visual heatmap showing where the model is looking.
    - **Surface Roughness**: Heuristic score based on edge density (smoother = more worn).

## Models
- `models/tire_wear_model.keras`: MobileNetV3 (Fine-tuned)
- `models/vehicle_model.keras`: ResNet50V2 (Fine-tuned)
