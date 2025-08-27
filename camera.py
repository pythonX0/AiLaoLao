import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import io

# ==============================================================================
# 1. Configuration and Model Loading
# ==============================================================================
# @st.cache_resource loads the model only once for efficiency.
@st.cache_resource
def load_model_and_processor():
    """
    Loads the pre-trained ViT model and feature extractor.
    """
    try:
        # Check for available device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"🖥️ Running on: **{device}**")

        # Load the pre-trained ViT feature extractor and model
        processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Define the class names for vehicle types
        class_names = ['Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 
                       'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 
                       'Taxi', 'Truck', 'Van']
        
        # Adjust the model's final layer to match the number of classes
        model.classifier = nn.Linear(model.config.hidden_size, len(class_names))
        
        # Path to the fine-tuned model weights
        model_path = './vit_vehicle_classifier_hf_dataset.pth'
        
        # Load the state dictionary from the saved model file
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set the model to evaluation mode
        
        return model, processor, class_names, device
    except FileNotFoundError:
        st.error(f"⚠️ Error: Model file not found at {model_path}. Please make sure the model has been trained and saved.")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None, None

# Load the model, processor, and class names
model, processor, class_names, device = load_model_and_processor()

# ==============================================================================
# 2. UI Layout
# ==============================================================================
# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Vehicle Type Recognition",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Type Recognition")

st.markdown(
    """
    Take a picture of a vehicle from your camera. The model will try to identify if it is an
    **'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 'Cart', 'Caterpillar', 
    'Helicopter', 'Limousine', 'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 
    'Taxi', 'Truck', 'Van'**.
    """
)

# ==============================================================================
# 3. Camera Input
# ==============================================================================
# Use the camera input component to capture a photo
camera_photo = st.camera_input("Take a picture")

if camera_photo is not None and model is not None:
    # Use a container to group elements
    with st.container():
        st.subheader("🖼️ Your Captured Image")
        # Convert the captured photo from BytesIO to a PIL Image
        image = Image.open(io.BytesIO(camera_photo.read())).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("---")

        st.subheader("🧠 Analyzing...")
        
        # ======================================================================
        # 4. Prediction
        # ======================================================================
        try:
            # Prepare the image for the model
            input_tensor = processor(images=image, return_tensors="pt").pixel_values
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                # Get the model's output
                output = model(input_tensor)
                logits = output.logits
                probabilities = F.softmax(logits, dim=1)
                
                # Get the predicted class and its confidence
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
            
            # Create a dataframe to show the probability for each class
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
        st.info("⬆️ Please take a picture with your camera to get started.")

