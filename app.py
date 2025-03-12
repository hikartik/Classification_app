from flask import Flask, request, render_template
from PIL import Image
import os
import torch
import torch.nn as nn
import timm  # pip install timm
from torchvision import transforms, models

# Initialize Flask application
app = Flask(__name__)

# Global device setup: use GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define number of classes and class names (adjust as needed)
num_classes = 6
class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Dictionary mapping model names (as displayed in the dropdown) to their weight file paths.
model_files = {
    "resnet18": "model/resnet18_intel.pth",
    "efficientnet": "model/efficientnet_intel.pth",
    "mobilenet": "model/mobilenet_intel.pth",
    "mobilevit": "model/mobilevit_intel.pth",
    "swin": "model/swin_intel.pth",
    "convnext": "model/convnext_intel.pth"
}

def get_model(model_name, num_classes, device):
    """
    Returns a model with its classifier head modified for num_classes.
    Supported models: 'convnext', 'efficientnet', 'mobilenet', 'mobilevit', 'resnet18', 'swin', and 'vit'.
    """
    if model_name == "convnext":
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilevit":
        model = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "swin":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Model name not recognized. Choose from 'convnext', 'efficientnet', 'mobilenet', 'mobilevit', 'resnet18', 'swin', 'vit'.")
    
    model = model.to(device)
    return model

# Define the prediction transform (should match training transforms)
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Home route: display the upload form with a dropdown for model selection
@app.route("/")
def index():
    available_models = list(model_files.keys())
    return render_template("index.html", models=available_models)

# Prediction endpoint: load the chosen model, process the uploaded image, and return prediction.
@app.route("/predict", methods=["POST"])
def predict():
    # Check for image file in request
    if "image" not in request.files:
        return render_template("index.html", prediction="No image file provided.", models=list(model_files.keys()))
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected.", models=list(model_files.keys()))
    
    # Get the selected model name from the form (default to 'resnet18' if not specified)
    selected_model = request.form.get("model_name", "resnet18")
    
    try:
        # Open and process the image file
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return render_template("index.html", prediction=f"Invalid image file: {str(e)}", models=list(model_files.keys()))
    
    img_tensor = predict_transform(img).unsqueeze(0).to(device)
    
    # Load the chosen model and its saved weights
    try:
        model = get_model(selected_model, num_classes, device)
        weights_path = model_files.get(selected_model)
        if not os.path.exists(weights_path):
            return render_template("index.html", prediction=f"Weight file not found for model {selected_model}.", models=list(model_files.keys()))
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict,strict=False)
        model.eval()
    except Exception as e:
        return render_template("index.html", prediction=f"Error loading model: {str(e)}", models=list(model_files.keys()))
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    
    return render_template("index.html", prediction=f"Predicted class: {predicted_class}", models=list(model_files.keys()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
