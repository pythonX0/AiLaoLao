import os
import streamlit as st
import google.generativeai as genai
import time

# --- Configuration and Setup ---
st.set_page_config(
    page_title="AI-LAO Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f4f6; /* Light gray background */
        color: #333;
        font-family: 'Phetsarath OT', sans-serif;
        .h1, .h2, .h3 {
            font-family: 'Phetsarath OT', sans-serif;
        }
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        font-family: 'Phetsarath OT', sans-serif;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .chat-container {
        border-radius: 15px;
        padding: 15px;
        background-color: #f3f4f6;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        margin-bottom: 20px;
        font-family: 'Phetsarath OT', sans-serif;
    }
    .user-message {
        background-color: #e0f7fa; /* Light cyan for user messages */
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
        font-family: 'Phetsarath OT', sans-serif;
    }
    .model-message {
        background-color: #f3f4f6; /* Lighter gray for model messages */
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Phetsarath OT', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get API key from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("⚠️ **ເກີດຂໍ້ຜິດພາດ**: ກະລຸນາກຳນົດ `GOOGLE_API_KEY` ໃນ environment variables.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Session State Initialization ---
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[
        {"role": "model", "parts": ["ສະບາຍດີ! ຂ້ອຍແມ່ນ SmartZ ຍິນດີໃຫ້ບໍລິການ. ທ່ານມີຫຍັງຢາກຖາມບໍ່?"]}
    ])
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main UI and Chat Logic ---
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🤖 Chatbot with SmartZ</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #555;'>AI ຜູ້ຊ່ວຍທີ່ພ້ອມຕອບທຸກຄຳຖາມຂອງທ່ານ</h3>",
    unsafe_allow_html=True
)

# Display a button to clear chat history
# if st.button("💬 ເລີ່ມການສົນທະນາໃໝ່"):
#     st.session_state.chat = model.start_chat(history=[
#         {"role": "model", "parts": ["ສະບາຍດີ! ຍິນດີຕອບຄຳຖາມຂອງທ່ານ."]}
#     ])
#     st.session_state.messages = []
#     st.experimental_rerun()

# --- Display Conversation History ---
# Use a custom message style
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='model-message'>{message['content']}</div>", unsafe_allow_html=True)

# --- Main Chat Input ---
if prompt := st.chat_input("💬 ຖາມມາເລີຍ! ຂ້ອຍຕອບໄດ້ຢູ່ແລ້ວ..."):
    # Add user message to display log
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

    # Get response from the Gemini API
    with st.spinner("ກຳລັງຄິດ..."):
        try:
            full_response = ""
            message_placeholder = st.empty()
            
            # Use streaming response
            for chunk in st.session_state.chat.send_message(prompt, stream=True):
                full_response += chunk.text
                message_placeholder.markdown(f"<div class='model-message'>{full_response}▌</div>", unsafe_allow_html=True)
            
            message_placeholder.markdown(f"<div class='model-message'>{full_response}</div>", unsafe_allow_html=True)

            # Add model response to display log
            st.session_state.messages.append({"role": "model", "content": full_response})

        except Exception as e:
            st.error(f"❌ **ເກີດຂໍ້ຜິດພາດ**: {e}")