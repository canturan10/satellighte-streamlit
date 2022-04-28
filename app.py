import numpy as np
import requests
import satellighte as sat
import streamlit as st
from PIL import Image


def main():
    # pylint: disable=no-member

    st.title("Satellighte Demo Page")
    st.write(
        "**Satellighte** is an image classification library  that consist state-of-the-art deep learning methods. It is a combination of the words **'Satellite'** and **'Light'**, and its purpose is to establish a light structure to classify satellite images, but to obtain robust results."
    )

    st.sidebar.title("Satellighte")
    url = "https://raw.githubusercontent.com/canturan10/satellighte/master/src/satellighte.png?raw=true"
    satellighte = Image.open(requests.get(url, stream=True).raw)
    st.sidebar.caption(f"Version `{sat.__version__}`")
    st.sidebar.image(satellighte, width=100)

    uploaded_file = st.sidebar.file_uploader(
        "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    st.sidebar.write(
        "[Find sample images on Satellighte.](https://github.com/canturan10/satellighte/tree/master/src/eurosat_samples/)"
    )

    selected_model = st.sidebar.selectbox(
        "Select model",
        sat.available_models(),
    )
    selected_version = st.sidebar.selectbox(
        "Select version",
        sat.get_model_versions(selected_model),
    )

    if uploaded_file is None:
        # Default image.
        url = "https://raw.githubusercontent.com/canturan10/satellighte/master/src/eurosat_samples/HerbaceousVegetation.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    st.write("### Inferenced Image")
    image = np.array(image.convert("RGB"))
    FRAME_WINDOW = st.image([], use_column_width=True)

    model = sat.Classifier.from_pretrained(selected_model, selected_version)
    model.eval()

    results = model.predict(image)
    pil_img = sat.utils.visualize(image, results)

    st.write("Results:", results)
    FRAME_WINDOW.image(pil_img)


if __name__ == "__main__":
    main()
