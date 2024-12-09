import streamlit as st

# Set page title and layout
st.set_page_config(
    page_title="Streamlit Multi-page App",
    layout="centered",
)

# Create a welcome header for the main page
st.title("Welcome to the Multi-Page Streamlit App")
st.write(
    """
    This app contains multiple sections:
    - **Introduction & Documentation**: A detailed description of the app.
    - **Model Inference**: Run inference using a pre-trained model.
    """
)

# Navigation instruction
st.write("Use the sidebar to navigate through the different pages.")
