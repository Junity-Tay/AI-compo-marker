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
        

def get_feedback():
    prompt = f"""
You are a PSLE English Composition Examiner.

Mark based on this rubric:
Band 5, 16-18: Content is fully relevant, highly interesting and thoroughly developed. Grammar, expression, spelling and punctuation are used accurately. There is a wide range of vocabulary and structures appropriately used. There is evidence of excellent sequencing, paragraphing and linkage of ideas.
Band 4, 13-15: Content is relevant, interesting and well developed. Grammar, expression, spelling and punctuation are mostly accurate. There is an adequate range of vocabulary and structures used mostly appropriately. There is evidence of good sequencing, paragraphing and linkage of ideas.
Band 3, 9-12: Content is generally relevant, fairly interesting and sufficiently developed. Grammar, expression, spelling and punctuation are used with some accuracy. There is fairly adequate range of vocabulary and structures used. There is evidence of fairly good sequencing, paragraphing and linkage of ideas.
Band 2, 5-8: Content has some relevance, fair attempts to be interesting and minimally developed. Grammar, expression, spelling and punctuation are used with varying degrees of accuracy. Simple vocabulary and structures are used. There are some attempts at sequencing, paragraphing and linkage of ideas.
Band 1, 1-4: Content has little relevance, slight attempts to be interesting and developed. Few instances of correct use of grammar, expression, spelling and punctuation. Few instances of simple vocabulary and structures. There is slight attempt at sequencing, paragraphing and linkage of ideas.

Give feedback based on these criteria:
1. Grammar: The student should display evidence of proper grammer use.
2. Expression: The student should display evidence of sentences with strong vocabulary to provide reader a clearer picture or atmosphere of the scenario.
3. Spelling: The student should display evidence of correct spelling.
4. Punctuation: The student should display evidence of punctuation used correctly.
5. Content: The student should display evidence of coherent and logical flow of events in the story so reader is able to follow and understand.

Give feedback in this format:
1. Estimated Band:
2. Estimated Marks /40:
3. Strengths:
4. Areas to improve or correct:
5. Improved Sample Submission:
6. Short Encouragement:
"""
    response = client.responses.create(
        model=st.secrets.get("OPENAI_MODEL", "gpt-5.6-luna"),
        input=prompt
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

    st.divider()

    if st.button("Get Composition Feedback"):
        with st.spinner("Marking in progress..."):
            feedback = get_feedback()
        st.subheader("Compo Feedback")
        st.write(feedback)
