import streamlit as st
import os
import tempfile
import google.generativeai as genai
import time
import fitz  # PyMuPDF for PDF rendering
import io
from PIL import Image
import requests

# Page configuration
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
GEMINI_API_KEY = "AIzaSyDoVdAvZXiHVdzZck30JtUkTXBmbvPJgJU"

# Set up base prompt
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

# Additional prompt for when old USPs are provided
old_usps_prompt = """
Additionally, I'm providing you with a list of previously identified USPs for this or a similar property. Review these old USPs and consider them alongside the brochure contents.

OLD USPs:
{old_usps}

Please merge the insights from both sources, removing duplicates and preserving the most compelling and unique selling points from both the brochure and the old USP list. Ensure your final list represents the strongest combined USPs, maintaining all the formatting and length requirements mentioned earlier.
"""

# Main content area
st.write("Upload Brochure or Enter URL and (Optionally) Enter Old USPs")

# File uploader
uploaded_file = st.file_uploader("Upload a brochure file", type=["pdf"])

# URL input
st.write("OR")
pdf_url = st.text_input("Enter URL to PDF brochure", placeholder="https://example.com/brochure.pdf")

# Text area for old USPs
st.subheader("Enter Old USPs (Optional)")
old_usps = st.text_area("Paste previous USPs here", height=200)

def setup_gemini_api():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False

def download_pdf_from_url(url):
    """Download a PDF from a URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Check if the content is actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            st.error(f"The URL does not point to a valid PDF file. Content-Type: {content_type}")
            return None
        
        # Return the content as bytes
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None

def analyze_pdf(pdf_bytes, prompt):
    """Analyze PDF directly with Gemini"""
    try:
        with st.spinner("Analyzing brochure with AI..."):
            # Initialize the Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Create a temporary file to store the PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(pdf_bytes)
            temp_file.close()
            
            # Upload the PDF file directly
            uploaded_pdf = genai.upload_file(temp_file.name, mime_type="application/pdf")
            
            # Generate content using Gemini with the PDF
            response = model.generate_content([prompt, uploaded_pdf])
            
            # Clean up the temporary file
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as e:
                pass
                
            return response.text

    except Exception as e:
        st.error(f"Error generating content with Gemini: {str(e)}")
        return None

def render_pdf_preview(pdf_bytes):
    """Render the first page of a PDF as an image for preview"""
    try:
        # Create a memory buffer from the PDF bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(pdf_document) > 0:
            # Load the first page
            first_page = pdf_document.load_page(0)
            
            # Render page to an image with a reasonable resolution
            pix = first_page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Close the document
            pdf_document.close()
            
            return img
        else:
            pdf_document.close()
            return None
    except Exception as e:
        st.error(f"Error rendering PDF preview: {str(e)}")
        return None

# Check if we have a PDF from either upload or URL
pdf_bytes = None
input_source = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer for later use
    input_source = f"File: {uploaded_file.name}"
elif pdf_url and pdf_url.strip():
    pdf_bytes = download_pdf_from_url(pdf_url.strip())
    if pdf_bytes:
        input_source = f"URL: {pdf_url}"

if pdf_bytes:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Brochure")
        st.write(input_source)
        
        # Display the first page as a preview
        preview_image = render_pdf_preview(pdf_bytes)
        if preview_image:
            st.image(preview_image, caption="First page preview", use_container_width=True)
        else:
            st.warning("Could not generate preview for this PDF")
    
    with col2:
        st.subheader("Property USPs")
        
        analyze_button = st.button("Extract USPs")
        
        # For better UX, show a placeholder immediately
        result_placeholder = st.empty()
        
        if analyze_button:
            if not setup_gemini_api():
                st.stop()
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Show thinking state to user
            result_placeholder.info("Thinking... This may take 30-60 seconds depending on the file size.")
            
            # Prepare the prompt based on whether old USPs are provided
            full_prompt = base_prompt
            if old_usps.strip():
                full_prompt += old_usps_prompt.format(old_usps=old_usps)
            
            # Analyze PDF directly
            analysis = analyze_pdf(pdf_bytes, full_prompt)
            
            # Display execution time
            execution_time = time.time() - start_time
            
            if analysis:
                # Clear placeholder and show result
                result_placeholder.empty()
                st.markdown(analysis)
                st.caption(f"Analysis completed in {execution_time:.1f} seconds")
                
                # Option to download results
                st.download_button(
                    label="Download USPs",
                    data=analysis,
                    file_name="property_usps.txt",
                    mime="text/plain"
                )
            else:
                result_placeholder.error("Failed to generate analysis. Please try again.")

# Footer
st.divider()
st.caption("Premium Property USP Analyzer - Powered by Google Gemini")


