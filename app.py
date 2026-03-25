import streamlit as st
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torchvision.transforms as transforms

st.set_page_config(page_title="Image Captioning AI", page_icon="📸", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF6666;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .caption-box {
        background-color: #1E2127;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        font-size: 20px;
        font-weight: 500;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #8892B0;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pretrained_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, feature_extractor, tokenizer

def predict_pretrained(image, model, feature_extractor, tokenizer):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()
st.markdown("<h1>Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform your images into descriptive stories using the power of Deep Learning.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button("✨ Generate Caption ✨", use_container_width=True)
    
    if generate_btn:
        with st.spinner("Analyzing visual features and generating sequence..."):
            model, feature_extractor, tokenizer = load_pretrained_model()
            caption = predict_pretrained(image, model, feature_extractor, tokenizer)
        
        st.markdown(f"""
            <div class="caption-box">
                📝 <b>Generated Caption:</b><br><br>
                <i style="color: #FFDE59;">"{caption.capitalize()}"</i>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
