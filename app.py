import random
from datetime import datetime

import numpy as np
import requests
import satellighte as sat
import streamlit as st
from PIL import Image


def main():
    # pylint: disable=no-member

    st.set_page_config(
        page_title="Satellighte Demo Page",
        page_icon="📡",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://canturan10.github.io/satellighte/",
            "About": "Satellite Image Classification",
        },
    )

    st.title("Satellighte Demo Page")

    url = "https://raw.githubusercontent.com/canturan10/satellighte/master/src/satellighte.png?raw=true"
    satellighte = Image.open(requests.get(url, stream=True).raw)

    st.sidebar.image(satellighte, width=100)
    st.sidebar.title("Satellighte")
    st.sidebar.caption(sat.__description__)

    st.write(
        "**Satellighte** is an image classification library  that consist state-of-the-art deep learning methods. It is a combination of the words **'Satellite'** and **'Light'**, and its purpose is to establish a light structure to classify satellite images, but to obtain robust results."
    )

    st.sidebar.caption(f"Version: `{sat.__version__}`")
    st.sidebar.caption(f"License: `{sat.__license__}`")
    st.sidebar.caption(sat.__copyright__)

    selected_model = st.selectbox(
        "Select model",
        sat.available_models(),
    )
    selected_version = st.selectbox(
        "Select version",
        sat.get_model_versions(selected_model),
    )

    model = sat.Classifier.from_pretrained(selected_model, selected_version)
    model.eval()

    uploaded_file = st.file_uploader(
        "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    if uploaded_file is None:
        st.write("Sample Image")
        # Sample image.
        url = f"https://raw.githubusercontent.com/canturan10/satellighte/master/src/eurosat_samples/{random_sample}?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    image = np.array(image.convert("RGB"))
    FRAME_WINDOW = st.image([], use_column_width=True)

    model = sat.Classifier.from_pretrained(selected_model, selected_version)
    model.eval()

    results = model.predict(image)
    pil_img = sat.utils.visualize(image, results)

    st.write("Results:", results)
    FRAME_WINDOW.image(pil_img)


if __name__ == "__main__":
    samples = [
        "AnnualCrop.jpg",
        "Forest.jpg",
        "HerbaceousVegetation.jpg",
        "PermanentCrop.jpg",
        "River.jpg",
    ]
    random.seed(datetime.now())
    random_sample = samples[random.randint(0, len(samples) - 1)]

    main()
