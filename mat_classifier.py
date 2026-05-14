import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image


# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(idx_to_class):

    model = models.efficientnet_v2_s(pretrained=False)

    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(

        nn.Dropout(0.5),

        nn.Linear(num_features, 512),

        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(512, len(idx_to_class))
    )

    model.load_state_dict(

        torch.load(
            "lepto_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model


# =====================================================
# PREDICTION
# =====================================================

def predict(model, image_path, idx_to_class):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    class_name = idx_to_class[
        predicted.item()
    ]

    return (
        class_name,
        confidence.item(),
        image
    )