import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_NAMES = [
    "Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", 
    "Nevus", "Pigmented Benign Keratosis", "Seborrheic Keratosis", 
    "Squamous Cell Carcinoma", "Vascular Lesion"
]

def clear_static_folder():
    if not os.path.exists('static'):
        os.makedirs('static')
    for filename in os.listdir('static'):
        file_path = os.path.join('static', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

model_yolo = YOLO('best28.pt')
model_classification = models.efficientnet_b7(weights=None)
num_ftrs = model_classification.classifier[1].in_features
model_classification.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, len(CLASS_NAMES))
)
model_classification.load_state_dict(torch.load("my_model.pth", map_location="cpu"), strict=False)
model_classification.eval()

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure 3-channel image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def predict(image):
    image_path = os.path.join('static', 'uploaded_image.png')
    image.save(image_path)
    results = model_yolo.predict(source=image_path, save=False)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_box_idx = confidences.argmax()
        best_result = results[0]
        best_result.boxes = best_result.boxes[best_box_idx:best_box_idx+1]
        result_image_np = best_result.plot()
        label = best_result.names[int(best_result.boxes.cls[0].item())]
    else:
        result_image_np = cv2.imread(image_path)
        label = "No detection"
    result_image_rgb = cv2.cvtColor(result_image_np, cv2.COLOR_BGR2RGB)
    result_image_pil = Image.fromarray(result_image_rgb)
    static_image_path = os.path.join('static', 'result_image.png')
    result_image_pil.save(static_image_path)
    return static_image_path, label

def generate_gradcam(image):
    # Preprocess the image to the correct format
    img_tensor = preprocess_image(image)

    # For EfficientNet, use the last convolutional layer
    target_layers = [model_classification.features[-1]]  # The last convolutional block

    # Initialize GradCAM
    cam = GradCAM(model=model_classification, target_layers=target_layers)

    # Generate the Grad-CAM image
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))

    # Convert the input image to numpy format (resize to 224x224 if needed)
    image_resized = image.resize((224, 224))

    # Ensure the image is in RGB format, even if the input has transparency (RGBA)
    image_rgb = image_resized.convert("RGB")
    image_np = np.array(image_rgb)

    # Normalize the image between 0 and 1 (for blending with heatmap)
    image_np = image_np / 255.0

    # Apply Grad-CAM to the image
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Save the Grad-CAM image
    gradcam_path = os.path.join('static', 'gradcam.png')
    plt.imsave(gradcam_path, cam_image)

    return gradcam_path



st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; color: #2C3E50;">Skin Cancer Detection</h1>', unsafe_allow_html=True)

st.write("""
    <div style="text-align: center; color: #555; font-size: 18px;">
    This deep learning model detects skin cancer and identifies various skin lesion types.
    </div>
""", unsafe_allow_html=True)

st.warning("Disclaimer: This model is a prototype ")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        clear_static_folder()
        result_image_path, label = predict(image)
        st.image(result_image_path, caption="Predicted Image", use_container_width=True)
        st.markdown(f"<h3 style='text-align: center; color: #E74C3C;'>Predicted Label: {label}</h3>", unsafe_allow_html=True)
        
        gradcam_path = generate_gradcam(image)
        st.image(gradcam_path, caption="Grad-CAM Visualization", use_container_width=True)
        
        with open(result_image_path, "rb") as file:
            st.download_button(
                label="Download Predicted Image",
                data=file,
                file_name="predicted_image.png",
                mime="image/png"
            )