import streamlit as st

st.set_page_config(page_title="AI-LAO-LAO", page_icon="🗺️")

# Use a custom title with markdown for better styling
st.markdown("<h1 style='text-align: center; color: #ff5733;'>Welcome to AI-LAO-LAO!</h1>", unsafe_allow_html=True)

st.write("---")


st.image("./logo1.png", use_container_width=True)

st.markdown("""
<br>
This app demonstrates the power of multi-AI applications.
You can navigate to different pages using the sidebar on the left.

---

### **Pages:**
- **Vehicle Type Recognition:** Upload an image and predict the type of vehicle.
- **SmartZ:** Chat anything with our Chatbot using the powerful Gemini API.
- **WhatsFood:** Upload an image of food and get a recipe suggestion.            

<br>
""", unsafe_allow_html=True)

# Add some visual flair at the bottom
st.balloons()