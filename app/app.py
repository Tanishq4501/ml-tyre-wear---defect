import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import pandas as pd
from ultralytics import YOLO

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import TireAnalyzer

# --- Page Config ---
st.set_page_config(page_title="Tire Inspection AI", layout="wide")

st.title("AI Tire Damage & Quality Inspection")
st.markdown("### Upload tire images to predict wear and analyze surface texture.")

# --- Load Models Once ---
@st.cache_resource
def load_analyzer():
    analyzer = TireAnalyzer()
    return analyzer

analyzer = load_analyzer()

@st.cache_resource
def load_yolo_model():
    try:
        # Use relative path from app/app.py (root/app/app.py -> root/models/...)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'yolov8n-seg-best.pt')
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

yolo_model = load_yolo_model()

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# --- Main Page Header ---
st.divider()

@st.cache_resource
def load_analyzer():
    return TireAnalyzer()

analyzer = load_analyzer()

@st.cache_resource
def load_yolo_model():
    # Load the trained segmentation model
    # Adjust path if needed based on where 'best.pt' is saved
    model_path = r"F:\TechS\ML_Tire\runs\segment\tire_defect_segmentation\weights\best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning(f"Feature Disabled: Defect Model not found at {model_path}")
        return None

yolo_model = load_yolo_model()

if not uploaded_files:
    # Empty State Dashboard
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Welcome</h2>
        <p style='font-size: 18px; color: #b0b3b8;'>
            Upload tire images in the sidebar to begin the automated inspection.
        </p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Analyzing Image {idx+1}: {uploaded_file.name}")
        
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") 
        tfile.write(uploaded_file.getbuffer())
        tfile.flush()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption=f'Image {idx+1}', use_column_width=True)
            
        with col2:
                try:
                    st.info(f"Processing {uploaded_file.name}...")
                    
                    # Inference
                    original_img, input_tensor = analyzer.preprocess_image(tfile.name)
                    
                    wear_class, wear_conf = analyzer.predict_wear(input_tensor)
                    vehicle_class, vehicle_conf = analyzer.predict_vehicle(input_tensor)
                    
                    # Advanced Metrics
                    edge_density = analyzer.calculate_edge_density(original_img)
                    est_depth_mm, groove_density = analyzer.estimate_tread_depth(original_img)
                    tread_coverage = analyzer.calculate_tread_coverage(original_img)
                    
                    # --- Defect Segmentation (YOLOv8) ---
                    defect_plot = None
                    defect_count = 0
                    
                    if yolo_model:
                        # Reset pointer for PIL Image
                        uploaded_file.seek(0)
                        img_for_yolo = Image.open(uploaded_file)
                        results = yolo_model(img_for_yolo)
                    
                        # Plot the results
                        defect_plot = results[0].plot() 
                        defect_plot = cv2.cvtColor(defect_plot, cv2.COLOR_BGR2RGB)
                        
                        defect_count = len(results[0].boxes)
                        
                        # --- Exact Detailed Parameters ---
                        defect_details = []
                        
                        CLASS_MAPPING = {
                            "tr": "Tread Damage",
                            "CBU": "Casing Break-Up",
                            "bead_damage": "Bead Damage",
                            "cut": "Cut/Puncture"
                        }

                        if results[0].masks is not None:
                            img_area = results[0].orig_shape[0] * results[0].orig_shape[1]
                            
                            for i, box in enumerate(results[0].boxes):
                                cls_id = int(box.cls[0])
                                raw_label = yolo_model.names[cls_id]
                                label = CLASS_MAPPING.get(raw_label, raw_label)
                                conf = float(box.conf[0])
                                
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                width = x2 - x1
                                height = y2 - y1
                                box_area = width * height
                                
                                defect_details.append({
                                    "Type": label,
                                    "Confidence": f"{conf:.2%}",
                                    "Width (px)": f"{width:.0f}",
                                    "Height (px)": f"{height:.0f}",
                                    "Box Area (px²)": f"{box_area:.0f}",
                                    "Rel. Size (%)": f"{(box_area / img_area):.2%}"
                                })

                    # --- Insight & Analysis Logic ---
                    health_score = 100
                    recommendations = []
                    
                    # Deduct for Wear
                    if wear_class == "Three Quarter Worn":
                        health_score -= 40
                        recommendations.append("Tire is nearing end of life. Plan for replacement.")
                    elif wear_class == "Fully Worn":
                        health_score -= 80
                        recommendations.append("Tread is dangerously low. **Replace Immediately.**")
                    
                    # Deduct for Defects
                    for defect in defect_details:
                        rel_size = float(defect["Rel. Size (%)"].strip('%'))
                        dtype = defect["Type"]
                        
                        severity = "Minor"
                        if rel_size > 5.0 or dtype in ["Casing Break-Up", "Cut/Puncture"]:
                            severity = "Critical"
                            health_score -= 50
                            recommendations.append(f"Critical {dtype} detected! Unsafe to drive.")
                        elif rel_size > 1.0:
                            severity = "Moderate"
                            health_score -= 20
                            recommendations.append(f"Monitor {dtype} closely. Repair if possible.")
                        else:
                            health_score -= 5
                        
                        defect["Severity"] = severity

                    health_score = max(0, health_score)

                    # --- Display Results ---
                    st.success(f"Analysis Complete for {uploaded_file.name}")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric(label="Predicted Wear", value=wear_class.upper(), delta=f"{wear_conf*100:.1f}% Conf")
                    with m2:
                        st.metric(label="Est. Tread Depth", value=f"{est_depth_mm:.1f} mm", delta="Needs Calibration", delta_color="off")
                    with m3:
                        color = "normal" if health_score > 70 else "off" if health_score > 40 else "inverse"
                        st.metric(label="Health Score", value=f"{health_score}/100", delta=f"-{100-health_score} Loss")
                    with m4:
                        st.metric(label="Defects", value=defect_count, delta="None" if defect_count==0 else "Action Req.", delta_color="inverse")

                    with st.expander("See Advanced Geometric Metrics"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Surface Analysis**")
                            st.write(f"- Groove Density: {groove_density*100:.1f}%")
                            st.write(f"- Tread Pattern Coverage: {tread_coverage:.1f}%")
                            st.write(f"- Edge Density (Roughness): {edge_density:.4f}")
                        with c2:
                            st.markdown("**Calibration Note**")
                            st.info("Depth estimation is heuristic (1.5mm - 8.0mm range) based on groove darkness and width. Real-world accuracy requires a reference object (coin/ruler) in the image.")

                    if defect_count > 0:
                        st.error(f"⚠️ **{defect_count} Defect(s) Detected!**")
                        
                        # Show Detailed Metrics Table
                        df_defects = pd.DataFrame(defect_details)
                        # Reorder columns
                        st.dataframe(df_defects[["Type", "Severity", "Rel. Size (%)", "Confidence", "Width (px)", "Height (px)"]], use_container_width=True)
                        
                    else:
                        st.success("✅ No structural defects detected.")
                    
                    # Recommendations
                    if recommendations:
                        st.subheader("AI Recommendations")
                        for rec in recommendations:
                            if "Critical" in rec or "Replace" in rec:
                                st.error(f"• {rec}")
                            else:
                                st.warning(f"• {rec}")

                    # Visuals
                    st.markdown("### Defect Analysis")
                    if defect_plot is not None:
                        st.image(defect_plot, caption=f"Segmentation Mask ({defect_count} defects)", use_column_width=True)
                    else:
                        st.info("No segmentation result available.")

                    # Explainability
                    st.markdown("#### Model Focus (Grad-CAM)")
                    try:
                        heatmap = analyzer.get_gradcam(input_tensor, analyzer.wear_model)
                        if heatmap is not None:
                            # Resize original image to match heatmap for overlay
                            original_resized = cv2.resize(original_img, (224, 224))
                            
                            # Apply ColorMap to make it 3-channel (224, 224, 3)
                            heatmap = np.uint8(255 * heatmap)
                            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            
                            superimposed_img = heatmap * 0.4 + original_resized * 0.6
                            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
                            st.image(superimposed_img, channels="BGR", use_column_width=True, caption="Heatmap: Where the model is looking")
                        else:
                            st.warning("Grad-CAM unavailable (internal layer mismatch).")
                    except Exception as e:
                        st.error(f"Could not generate Grad-CAM: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {uploaded_file.name}: {e}")

        # Cleanup temp file
        tfile.close()
        try:
            os.remove(tfile.name)
        except PermissionError:
            pass
            
        st.markdown("---") # Separator between images
