import base64
import streamlit as st
from openai import OpenAI
from PIL import Image

st.title("Composition Text Extraction with OpenAI OCR")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_text(image_bytes, mime_type):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract all handwritten text from this composition image."
                            "Return only the extracted text. Preserve the paragraph and line breaks."
                            "Do not correct spelling, grammar, punctuation, or wording."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                        "detail":"original",
                    },
                ],
            }
        ],
    )

    return response.output_text
        

uploaded_image = st.file_uploader(
    "Upload your composition",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    image_bytes = uploaded_image.getvalue()
    mime_type = uploaded_image.type or "image/png"
    
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Composition", width="stretch")

    with st.spinner("Reading handwriting / text with OpenAI..."):
        extracted_text = extract_text(image_bytes, mime_type)

    st.subheader("Extracted Text:")
    st.text_area("OCR Result", extracted_text, height=300)
