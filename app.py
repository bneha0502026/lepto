import streamlit as st
from PIL import Image
import torch

from mat_classifier import *

st.set_page_config(
    page_title="Leptospira MAT Vision",
    layout="centered"
)

st.title("Leptospira MAT Vision")

st.write("Upload MAT image for prediction")


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image.save("temp_image.jpg")


    # =====================================================
    # CLASS LABELS
    # =====================================================

    idx_to_class = {

        0: '20-40%',

        1: '40-50%',

        2: '50-60%',

        3: '60-70%',

        4: 'Below 20%',

        5: 'More than 70%'
    }


    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = load_model(idx_to_class)


    # =====================================================
    # PREDICTION
    # =====================================================

    class_name, confidence, img = predict(

        model,

        "temp_image.jpg",

        idx_to_class
    )


    # =====================================================
    # POSITIVE / NEGATIVE CLASSES
    # =====================================================

    positive_classes = [

        "50-60%",

        "60-70%",

        "More than 70%"
    ]


    # =====================================================
    # FINAL RESULT
    # =====================================================

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


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.image(

        img,

        caption="Uploaded Image",

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