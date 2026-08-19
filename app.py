import streamlit as st
import numpy as np
import cv2 
from paddleocr import PaddleOCR
from PIL import Image

st.title("Composition Text Extraction with PaddleOCR")

@st.cache_resource
def load_ocr():
  return PaddleOCR(use_angle_cls=True, lang="en")

ocr = load_ocr()

uploaded_image = st.file_uploader("Upload your composition", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
  image = Image.open(uploaded_image)
  img_array = np.array(image)

  result = reader.readtext(img_array)

st.subheader("Extracted Text:")
extracted_text = ""
for detection in result:
  extracted_text += detection[1] + "\n"
st.text(extracted_text)

for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    img_cv2 = cv2.rectangle(img_cv2, top_left, bottom_right, (0, 255, 0), 3)
