import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from utils.model import get_model
from utils.prompt_analyzer import PromptAnalyzer

# Set page config
st.set_page_config(
    page_title="Jailbreak Detection",
    page_icon="🔒",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .risk-high { color: #ff0000; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #ffff00; font-weight: bold; }
    .risk-safe { color: #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    try:
        model = get_model()
        model_path = "models/best_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            st.session_state.model = model
        else:
            st.error("Model file not found. Please ensure the model file exists at models/best_model.pth")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if 'prompt_analyzer' not in st.session_state:
    st.session_state.prompt_analyzer = PromptAnalyzer()

def preprocess_image(image):
    try:
        # Convert image to grayscale
        image = image.convert('L')
        # Resize image to 32x32
        image = image.resize((32, 32))
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        # Add batch dimension and channel dimension
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(image):
    try:
        with torch.no_grad():
            output = st.session_state.model(image)
            prediction = (torch.sigmoid(output) > 0.5).item()
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def refine_prompt(image):
    try:
        with torch.no_grad():
            refined_prompt = st.session_state.model(image)
        return refined_prompt.item()
    except Exception as e:
        st.error(f"Error during prompt refinement: {str(e)}")
        return None

# Main UI
st.title("🔒 Jailbreak Detection and Prompt Refinement")
st.markdown("""
This application helps detect adversarial images and refine prompts for better security.
Upload an image to get started.
""")

# Create tabs
tab1, tab2 = st.tabs(["Image Analysis", "Prompt Analysis"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            preprocessed_image = preprocess_image(image)
            
            if preprocessed_image is not None:
                with col2:
                    if st.button("🔍 Detect Adversarial", key="detect"):
                        prediction = predict(preprocessed_image)
                        if prediction is not None:
                            if prediction == 1:
                                st.error("⚠️ The image is adversarial.")
                            else:
                                st.success("✅ The image is clean.")
                    
                    if st.button("✨ Refine Prompt", key="refine"):
                        refined_prompt = refine_prompt(preprocessed_image)
                        if refined_prompt is not None:
                            st.info(f"Refined Prompt: {refined_prompt:.4f}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.markdown("""
    This section analyzes text prompts for potential adversarial content.
    Enter a prompt to check for security risks.
    """)
    
    prompt = st.text_area("Enter your prompt:", height=150)
    
    if st.button("🔍 Analyze Prompt", key="analyze_prompt"):
        if prompt:
            risk_level, issues = st.session_state.prompt_analyzer.analyze_prompt(prompt)
            risk_description = st.session_state.prompt_analyzer.get_risk_description(risk_level)
            
            st.markdown(f"### Risk Level: <span class='risk-{risk_level}'>{risk_level.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**{risk_description}**")
            
            if issues:
                st.markdown("### Issues Found:")
                for issue in issues:
                    st.markdown(f"- {issue}")
            
            st.markdown("### Recommendations:")
            if risk_level == 'high':
                st.error("⚠️ This prompt should be rejected as it contains multiple adversarial patterns.")
            elif risk_level == 'medium':
                st.warning("⚠️ This prompt should be reviewed carefully before processing.")
            elif risk_level == 'low':
                st.info("ℹ️ This prompt may need minor modifications before processing.")
            else:
                st.success("✅ This prompt appears to be safe for processing.")
        else:
            st.warning("Please enter a prompt to analyze.")