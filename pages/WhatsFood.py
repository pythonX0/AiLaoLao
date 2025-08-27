import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Food AI", page_icon="🍔", layout="wide")

# Custom CSS for a luxurious, dark UI
st.markdown("""
<style>
    /* Import Google Fonts and Phetsarath OT font */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;700&family=Playfair+Display:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;700&display=swap');

    /* Main container styling */
    .main-container {
        padding: 2rem;
        background-color: #121212;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            font-family: 'Phetsarath OT', 'Playfair Display', serif;
    }
    
    /* Custom Title with Playfair Display for English and Phetsarath OT for Lao */
    .main-title {
        color: #F9C74F;
        font-family: 'Phetsarath OT', 'Playfair Display', serif;
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: 700;
        font-size: 3rem;
    }
    
    /* Custom Subtitle with Dancing Script and Phetsarath OT */
    .subtitle {
        color: #B0B0B0;
        font-family: 'Phetsarath OT', 'Dancing Script', cursive;
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 2em;
            
    }

    /* Button Styling */
    .stButton>button {
        background-color: #F9C74F;
        color: #121212;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-family: 'Phetsarath OT', 'Playfair Display', serif;
    }

    .stButton>button:hover {
        background-color: #F8B400;
        transform: scale(1.05);
    }
    
    /* General text styling with Phetsarath OT for Lao */
    .stText, .stMarkdown, .stSubheader {
        color: #F5F5F5;
        font-family: 'Phetsarath OT', sans-serif;
    }
    
    /* Headers with Playfair Display and Phetsarath OT */
    h2 {
        color: #F9C74F;
        text-align: center;
        font-family: 'Phetsarath OT', 'Playfair Display', serif;
    }

    /* File uploader styling */
    .stFileUploader label {
        color: #B0B0B0;
    }

    /* Image hover effect */
    .stImage {
        transition: transform 0.3s ease-in-out;
        cursor: pointer;
    }

    .stImage:hover {
        transform: scale(1.02);
    }
    
    /* Specific styling for the generated recipe text to make it more readable */
    .recipe-text {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        line-height: 1.6;
        white-space: pre-wrap;
            font-family: 'Phetsarath OT', 'Playfair Display', serif;
    }
</style>
""", unsafe_allow_html=True)

# --- API Key and Model Configuration ---
# Get API key from environment variables (more secure than hardcoding)
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Please set the 'GOOGLE_API_KEY' environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Main UI Layout ---
#st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1 class='main-title'>🍔 ຢາກກິນຫຍັງເດ້............!!!</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>You can upload an image and let's deliver you a delicious food.</p>", unsafe_allow_html=True)

st.write("---") # A simple separator line

# Main content container
st.header("ອາຫານທີ່ທ່ານຢາກກິນ")
st.markdown("<p class='subtitle'>What's food you would like to eat?.</p>", unsafe_allow_html=True)

# Use columns for a better layout
col1, col2 = st.columns([1, 1.5])

with col1:
    # File uploader component
    uploaded_file = st.file_uploader("Upload an image of food...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # The button to check the food
    check_button = st.button("Check Food")

with col2:
    if check_button:
        if uploaded_file is not None:
            image_prompt = "What is the name of the food in this image? And tell me the ingredients and how to make it and a recipe of this food."

            with st.spinner("Analyzing image..."):
                try:
                    # Prepare the prompt with both the text and the image
                    prompt_with_image = [image_prompt, image]
                    
                    # Call the Gemini model
                    response = model.generate_content(prompt_with_image)
                    
                    st.subheader("Menu:")
                    st.markdown(f"<div class='recipe-text'>{response.text}</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload an image.")

st.markdown("</div>", unsafe_allow_html=True)
