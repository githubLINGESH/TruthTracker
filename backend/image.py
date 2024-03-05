import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Define the model architecture with a pre-trained ResNet
class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Freeze all the layers in the pre-trained model
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the final layer for binary classification (REAL vs. FAKE)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

# Load the model (make sure to load your trained model weights)
model = FakeImageDetector()
# Placeholder for loading your model weights (e.g., model.load_state_dict(torch.load('path_to_your_trained_weights.pth')))
# model.load_state_dict(torch.load('path_to_your_trained_weights.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Streamlit application
def main():
    st.title("Deepfake Detector")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect'):
            # Preprocess the uploaded image
            processed_image = preprocess_image(image)
            # Perform inference
            with torch.no_grad():
                outputs = model(processed_image)
                _, predicted = torch.max(outputs, 1)
                result_text = 'REAL' if predicted.item() == 0 else 'FAKE'
                st.write(f"The image is likely {result_text}.")

if __name__ == '__main__':
    main()