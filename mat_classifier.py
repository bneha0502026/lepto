import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)

from PIL import Image

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD MODEL
# =========================================================

def load_model(idx_to_class):

    model = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.DEFAULT
    )

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, len(idx_to_class))
    )

    state_dict = torch.load(
        "lepto_model.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    return model

# =========================================================
# PREDICTION
# =========================================================

def predict(model, image_path, idx_to_class):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    class_name = idx_to_class[predicted.item()]

    return (
        class_name,
        confidence.item(),
        image
    )