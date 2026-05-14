import streamlit as st
from PIL import Image
import torch

from mat_classifier import *

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Leptospira MAT Vision",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("Leptospira MAT Vision")

st.write(
    "Upload one or more MAT images for prediction"
)

# =========================================================
# MULTIPLE IMAGE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(

    "Choose images",

    type=["jpg", "jpeg", "png"],

    accept_multiple_files=True
)

# =========================================================
# CLASS LABELS
# =========================================================

idx_to_class = {

    0: '20-40%',

    1: '40-50%',

    2: '50-60%',

    3: '60-70%',

    4: 'Below 20%',

    5: 'More than 70%'
}

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(idx_to_class)

# =========================================================
# PREDICTIONS
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        # =================================================
        # LOAD IMAGE
        # =================================================

        image = Image.open(uploaded_file)

        image = image.convert("RGB")

        temp_path = "temp_image.jpg"

        image.save(temp_path)

        # =================================================
        # PREDICT
        # =================================================

        class_name, confidence, img = predict(

            model,

            temp_path,

            idx_to_class
        )

        # =================================================
        # POSITIVE CLASSES
        # =================================================

        positive_classes = [

            "50-60%",

            "60-70%",

            "More than 70%"
        ]

        # =================================================
        # FINAL RESULT
        # =================================================

        if class_name in positive_classes:

            final_result = (

                "Reactive - Sample is positive "

                "for Leptospira antibodies."
            )

        else:

            final_result = (

                "Non-Reactive - Sample is negative "

                "for Leptospira antibodies."
            )

        # =================================================
        # DISPLAY
        # =================================================

        st.image(

            img,

            caption=uploaded_file.name,

            use_container_width=True
        )

        st.write(

            f"### Percentage Reduction of Leptospira: {class_name}"
        )

        st.write(

            f"### Final Result: {final_result}"
        )

        st.write(

            f"### Prediction Certainty: {confidence:.2%}"
        )