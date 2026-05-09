import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Leptospira MAT Classifier")

st.title("Leptospira MAT Classification")
st.write("Upload a MAT image for prediction.")

# -----------------------------
# CLASS NAMES
# -----------------------------
classes = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "More than 70%"
]

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    model = models.efficientnet_v2_s(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

model = load_model()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MAT Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    class_name = classes[predicted.item()]
    confidence_value = confidence.item()

    # -----------------------------
    # FINAL RESULT
    # -----------------------------
    doubtful_classes = [
        "40-50%",
        "50-60%"
    ]

    positive_classes = [
        "60-70%",
        "More than 70%"
    ]

    if class_name in doubtful_classes:
        final_result = "Doubtful Result - Requires further confirmation."

    elif class_name in positive_classes:
        final_result = "Reactive - Sample is positive for Leptospira antibodies."

    else:
        final_result = "Non-Reactive - Sample is negative for Leptospira antibodies."

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    st.write(f"Percentage Reduction of Leptospira: {class_name}")

    st.write(f"Final Result: {final_result}")

    st.write(f"Prediction Certainty: {confidence_value:.2%}")