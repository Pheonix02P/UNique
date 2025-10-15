import streamlit as st
import os
import tempfile
import time
import requests
import io

import fitz  # PyMuPDF for PDF rendering
from PIL import Image

from google import genai
from google.genai import types

# ========== Streamlit / UI Setup ==========

st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini (via google-genai)")

# Don’t hardcode API key in code if possible; better via environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDm3sAZsBlO2UTIsD9oBetBYJrhhXXdipE")

base_prompt = """You are provided with the attached brochure for a premium residential project. Your task is to extract the unique selling propositions (USPs) that will positively influence potential buyer decisions, keeping in mind the expectations of buyers in this segment.

Focus on the following aspects, aligning with the expectations of a premium homebuyer:

Thematic and Architectural Uniqueness  
Facilities and Luxury Amenities  
Technology and Security Features  
Landscape and Environment  
Location Highlights  
Awards and Recognition  
Any Other Unique Features that enhance lifestyle, convenience, and security  

Important Guidelines:

Keep in mind that the attachment may contain noise, so filter out any irrelevant content.

Output the USPs as bullet points, ensuring each bullet point is 20 words or less.

Ensure each point provides factual details about the project based on the information available in the brochure.

If and only if the proper name of an architect, designer, builder, consultant, or developer is explicitly mentioned in the brochure, include it in the USPs. Do not use common nouns such as "designers" or "architect" without the presence of a proper noun.

Arrange the USPs in descending order of uniqueness and appeal, placing the most attractive first.

Give priority to factual details explicitly mentioned in the text, such as clubhouse size, project density, greenery percentage, etc.

Use a professional tone in your bullet points.

Do not include headers in the bullet points.

Ensure grammatical correctness and capitalize the first letters of proper nouns.

Focus on factual information, lifestyle appeal, and renowned names associated with the project.

Convert each point into a 75-character limit, without losing the factual data of the point
"""

old_usps_prompt = """
Additionally, I'm providing you with a list of previously identified USPs for this or a similar property. Review these old USPs and consider them alongside the brochure contents.

OLD USPs:
{old_usps}

Please merge the insights from both sources, removing duplicates and preserving the most compelling and unique selling points from both the brochure and the old USP list. Ensure your final list represents the strongest combined USPs, maintaining all the formatting and length requirements mentioned earlier.
"""

st.write("Upload Brochure or Enter URL and (Optionally) Enter Old USPs")

model_options = {
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
}
selected_model = st.selectbox("Select Gemini model", options=list(model_options.values()), index=0)

uploaded_file = st.file_uploader("Upload a brochure (PDF)", type=["pdf"])
st.write("OR")
pdf_url = st.text_input("Enter URL to PDF brochure", placeholder="https://example.com/brochure.pdf")

st.subheader("Enter Old USPs (Optional)")
old_usps = st.text_area("Paste previous USPs here", height=200)

# ========== Utility Functions ==========

def download_pdf_from_url(url: str) -> bytes | None:
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "application/pdf" not in ct and not url.lower().endswith(".pdf"):
            st.error(f"URL does not point to a PDF. Content-Type: {ct}")
            return None
        return resp.content
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")
        return None

def render_pdf_preview(pdf_bytes: bytes) -> Image.Image | None:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count < 1:
            doc.close()
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.open(io.BytesIO(pix.tobytes("ppm")))
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error rendering PDF: {e}")
        return None

def create_genai_client():
    # Create client with API key mode (Gemini Developer API) or Vertex AI mode if needed
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

def analyze_pdf_with_genai(pdf_bytes: bytes, prompt: str, model_id: str) -> str | None:
    try:
        client = create_genai_client()

        # Write PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(pdf_bytes)
            tmp_path = tf.name

        # Upload file (new API — no mime_type argument)
        uploaded_file = client.files.upload(file=tmp_path)

        # Build contents: first the prompt (string), then a small dict referencing the uploaded file
        contents = [
            prompt,
            {"uri": uploaded_file.uri}
        ]

        # Call the model
        response = client.models.generate_content(
            model=model_id,
            contents=contents
        )

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        # Many SDK responses expose .text — fall back to extracting textual parts if needed
        if hasattr(response, "text") and response.text:
            return response.text
        # fallback attempt to read structured output
        try:
            # adapt depending on response structure
            return response.output[0].content[0].text
        except Exception:
            return str(response)

    except Exception as e:
        st.error(f"GenAI error: {e}")
        return None


# ========== Main Logic ==========

pdf_bytes = None
input_source = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    input_source = f"File: {uploaded_file.name}"
elif pdf_url:
    pdf_bytes = download_pdf_from_url(pdf_url.strip())
    if pdf_bytes is not None:
        input_source = f"URL: {pdf_url.strip()}"

if pdf_bytes:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Brochure")
        st.write(input_source)
        preview = render_pdf_preview(pdf_bytes)
        if preview:
            st.image(preview, caption="First page preview", use_container_width=True)
        else:
            st.warning("Could not generate preview")

    with col2:
        st.subheader("Property USPs")
        st.info(f"Using model: {selected_model}")
        if st.button("Extract USPs"):
            start = time.time()
            st.spinner("Analyzing brochure…")

            full_prompt = base_prompt
            if old_usps.strip():
                full_prompt += old_usps_prompt.format(old_usps=old_usps.strip())

            analysis = analyze_pdf_with_genai(pdf_bytes, full_prompt, selected_model)
            elapsed = time.time() - start

            if analysis:
                st.markdown(analysis)
                st.caption(f"Completed in {elapsed:.1f}s")
                st.download_button("Download USPs", data=analysis, file_name="usps.txt", mime="text/plain")
            else:
                st.error("Failed to generate USPs. Try a smaller PDF or simpler prompt.")

# Footer
st.divider()
st.caption("Premium Property USP Analyzer — Powered by Google Gemini / GenAI")



