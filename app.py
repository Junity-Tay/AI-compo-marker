import streamlit as st
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

st.title("Composition Text Extraction with PaddleOCR")

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu"
    )

ocr = load_ocr()

uploaded_image = st.file_uploader(
    "Upload your composition",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Composition", width='stretch')

    with st.spinner("Reading handwriting / text with PaddleOCR..."):
        result = ocr.predict(img_array)

    extracted_lines = []

    for res in result:
        data = res.json

        # PaddleOCR usually stores recognised text here:
        if "res" in data and "rec_texts" in data["res"]:
            extracted_lines.extend(data["res"]["rec_texts"])
        elif "rec_texts" in data:
            extracted_lines.extend(data["rec_texts"])

    extracted_text = "\n".join(extracted_lines)

    st.subheader("Extracted Text:")
    st.text_area("OCR Result", extracted_text, height=300)
