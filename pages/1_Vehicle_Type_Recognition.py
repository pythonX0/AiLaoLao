import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import os
import io

# ==============================================================================
# 1. ການຕັ້ງຄ່າ ແລະ ການໂຫຼດໂມເດລ
# ==============================================================================
# @st.cache_resource ຈະໂຫຼດໂມເດລພຽງຄັ້ງດຽວ
@st.cache_resource
def load_model_and_processor():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layout="right"
        st.write(f"🖥️ Running on: **{device}**")

        processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        
        class_names = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 
           'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 
           'Taxi', 'Truck', 'Van']
        model.classifier = nn.Linear(model.config.hidden_size, len(class_names))
        
        model_path = './vit_vehicle_classifier_hf_dataset.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, processor, class_names, device
    except FileNotFoundError:
        st.error(f"⚠️ Error: Model file not found at {model_path}. Please make sure the model has been trained and saved.")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None, None

model, processor, class_names, device = load_model_and_processor()

# ==============================================================================
# 2. UI Layout
# ==============================================================================
st.set_page_config(
    page_title="Vehicle Type Recognition",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Type Recognition")


st.markdown(
    """
    Upload an image of a **'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 
           'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 
           'Taxi', 'Truck', 'Van'**.
    """
)

# ==============================================================================
# 3. Image Upload
# ==============================================================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Use a container to group elements
    with st.container():
        st.subheader("🖼️ Your Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("---")

        st.subheader("🧠 Analyzing...")
        
        # ======================================================================
        # 4. Prediction
        # ======================================================================
        try:
            input_tensor = processor(images=image, return_tensors="pt").pixel_values
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
                logits = output.logits
                probabilities = F.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item() * 100
            
            # Use columns for a cleaner layout of results
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Class", value=predicted_class)
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")

            st.markdown("---")
            st.subheader("📊 Probability Breakdown")
            
            prob_data = {
                'Vehicle Type': class_names,
                'Probability (%)': [f"{prob.item()*100:.2f}%" for prob in probabilities[0]]
            }
            st.dataframe(prob_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.info("Please make sure the uploaded file is a valid image (JPEG, PNG).")
else:
    if model is not None:
        st.info("⬆️ Please upload an image file to get started.")