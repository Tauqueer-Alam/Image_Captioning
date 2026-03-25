# Image Captioning AI 📸

Transform your images into descriptive stories using the power of Deep Learning. This project features a modern Web Application built with Streamlit and a Hugging Face pre-trained image captioning model (`nlpconnect/vit-gpt2-image-captioning`). It also includes source code to build and train your own custom Image Captioning model using a Vision Transformer (ViT) and LSTM! 

## Features
- **Modern Streamlit UI**: A beautifully crafted, responsive, and interactive user interface.
- **Pre-trained Model**: Uses state-of-the-art Hugging Face models (ViT + GPT-2) for generating highly accurate captions.
- **Custom Model Code**: `model.py` and `train.py` are included to help you learn and train your own ViT-LSTM Image Captioner from scratch.

## Project Structure
- `app.py` - The main Streamlit web application.
- `model.py` - PyTorch implementation of a custom ViT-LSTM image captioning architecture.
- `train.py` - A training loop template for the custom ViT-LSTM model.
- `requirements.txt` - Required Python dependencies.

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd "Projects - Image Captioning"
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application locally, execute the following command:

```bash
streamlit run app.py
```

The app will pop up in your default web browser. From there, you can upload any image (JPG, JPEG, PNG) and the AI will generate a descriptive caption for it!
