import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import gdown
import os

# ==============================================================================
# 1. Configuration and Model Loading
# ==============================================================================

# Use st.cache_resource to cache the model, preventing it from reloading on every interaction.
@st.cache_resource
def load_model_and_processor():
    """
    Loads the pre-trained ViT model and processor.
    Handles downloading the model weights from Google Drive.
    """
    try:
        # Determine the device to run the model on (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"🖥️ Running on: **{device}**")

        # Load the feature extractor and base model from Hugging Face
        processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

        # Define the list of vehicle class names
        class_names = [
            'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar',
            'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank',
            'Taxi', 'Truck', 'Van'
        ]
        
        # Replace the model's final classifier layer to match the number of classes
        model.classifier = nn.Linear(model.config.hidden_size, len(class_names))

        # ----------------------------------------------------------------------------
        # Download the model file from Google Drive if it does not exist
        # NOTE: You must have 'gdown' installed: pip install gdown
        # IMPORTANT: Replace 'YOUR_FILE_ID' with the actual Google Drive file ID.
        # ----------------------------------------------------------------------------
        model_path = 'vit_vehicle_classifier_hf_dataset.pth'
        if not os.path.exists(model_path):
            st.info("📥 Downloading model file... This may take a while.")
            # Replace 'YOUR_FILE_ID' with the actual ID of your file
            url = 'https://drive.google.com/uc?id=1Ci-14nPs5_UWN0vORnAOhQR53VCrQl23'
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Download complete!")

        # Load the saved state dictionary into the model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        return model, processor, class_names, device
    
    except FileNotFoundError:
        st.error("⚠️ Error: Model file not found. Please check your Google Drive link and ensure it's a valid ID.")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Please ensure you have a valid Google Drive file ID and the 'gdown' package is installed.")
        return None, None, None, None

# Load the model and other necessary components
model, processor, class_names, device = load_model_and_processor()

# ==============================================================================
# 2. UI Layout
# ==============================================================================

# Set the Streamlit page configuration
st.set_page_config(
    page_title="Vehicle Type Recognition",
    page_icon="�",
    layout="centered"
)

# Display the main title and introduction
st.title("🚗 Vehicle Type Recognition")
st.markdown(
"""
Upload an image of one of the following vehicle types: **Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar,
Helicopter, Limousine, Motorcycle, Segway, Snowmobile, Tank, Taxi, Truck, Van**.
"""
)
st.markdown("---")

# ==============================================================================
# 3. Image Upload and Prediction Logic
# ==============================================================================

# Create a file uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# Only proceed if a file is uploaded and the model was loaded successfully
if uploaded_file is not None and model is not None:
    with st.container():
        st.subheader("🖼️ Your Uploaded Image")
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("---")

        st.subheader("🧠 Analyzing...")

        try:
            # Prepare the image for the model
            input_tensor = processor(images=image, return_tensors="pt").pixel_values
            input_tensor = input_tensor.to(device)

            # Perform the prediction
            with torch.no_grad():
                output = model(input_tensor)
                logits = output.logits
                probabilities = F.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Get the predicted class and confidence score
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100

            # Display the results using Streamlit metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Class", value=predicted_class)
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")

            st.markdown("---")
            st.subheader("📊 Probability Breakdown")

            # Create and display a dataframe of all class probabilities
            prob_data = {
                'Vehicle Type': class_names,
                'Probability (%)': [f"{prob.item()*100:.2f}%" for prob in probabilities[0]]
            }
            st.dataframe(prob_data, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.info("Please make sure the uploaded file is a valid image (JPEG, PNG).")

# If no file is uploaded yet, show an informational message
else:
    if model is not None:
        st.info("⬆️ Please upload an image file to get started.")

