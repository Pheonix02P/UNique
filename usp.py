import streamlit as st
import os
import tempfile
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import io
import google.generativeai as genai
import time

# Enable caching
@st.cache_data
def cached_image_extraction(pdf_bytes):
    """Cached function to extract images from PDF bytes"""
    return extract_images_from_pdf_bytes(pdf_bytes)

# Page configuration
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
GEMINI_API_KEY = "AIzaSyCIR8-WfadSCfOZTr1PxJFXRzP5HbiE9IQ"

# Set up prompt
prompt = """You are provided with the attached brochure for a premium residential project. Your task is to extract the unique
selling propositions (USPs) that will positively influence potential buyer decisions, keeping in mind the expectations of buyers in this segment.
Focus on the following aspects, aligning with the expectations of a premium homebuyer:
•            Thematic and Architectural Uniqueness
•            Facilities and Luxury Amenities
•            Technology and Security Features  
•            Landscape and Environment  
•            Location Highlights  
•            Awards and Recognition
•            Any Other
 
Unique Features that enhance lifestyle, convenience, and security.
NOTE:
•            Keep in mind that the attachment may contain noise, so filter out any irrelevant content.
•            Output the USPs as bullet points, ensuring each bullet point is 20 words or less.
•            Ensure each point provides factual details about the project based on the information
available in the brochure.
*Important : 
* If and only if the proper name of an architect, designer, builder,consultant, or developer is explicitly mentioned in the brochure, include it in the USPs, Do not use common nouns such as designers or architect without the presence of a proper noun* •  Arrange them in descending order, with the most unique and attractive USP at the top. -  Give priority to factual details explicitly mentioned in the text, such as the size of the clubhouse, project density, and greenery. •  Use a professional tone in your bullet points. •  Do not include headers in the bullet points. •  Ensure grammatical correctness and capitalize the first letters of proper nouns. Focus on : (factual information, lifestyle appeal, and renowned names associated with the project). • Include unique points and factual information from the following reference points given to you.
"""

# Main content area
st.write("Upload Brochure.")

# File uploader
uploaded_file = st.file_uploader("Choose a brochure file", type=["pdf", "jpg", "jpeg", "png"])

def setup_gemini_api():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return False

def process_image(image_file):
    try:
        image = Image.open(image_file)
        # Resize large images to optimize processing speed
        max_size = 1200
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size/image.width, max_size/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def extract_images_from_pdf_bytes(pdf_bytes):
    """Extract images from PDF bytes with optimized settings"""
    images = []
    temp_file = None
    
    try:
        # Create a temporary file for the PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(pdf_bytes)
        temp_file.close()
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(temp_file.name)
        total_pages = len(doc)
        
        # Process all pages
        with st.spinner(f"Extracting all {total_pages} pages..."):
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                # Lower resolution for speed (1.2 instead of 2)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                # Optimize image size for API
                max_dim = 1200
                if img.width > max_dim or img.height > max_dim:
                    ratio = min(max_dim/img.width, max_dim/img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                    
                images.append(img)
        
        # Close the document explicitly
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []
    finally:
        # Clean up
        try:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            pass

def analyze_whole_brochure(images, prompt):
    """Analyze brochure images in optimized batches"""
    try:
        with st.spinner("Analyzing brochure with AI..."):
            # Initialize the Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Helper: convert image to byte stream and upload
            def image_to_part(image, index):
                # Create a temporary file from the image
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                image.save(temp_file, format="PNG")
                temp_file.close()  # Close the file to allow it to be used for upload
                
                # Now upload the temporary file
                return genai.upload_file(temp_file.name, mime_type="image/png", display_name=f"page_{index+1}.png")

            # Select representative images
            if len(images) > 8:
                first_batch = images[:3]
                remaining = images[3:]
                num_remaining_samples = min(13, len(remaining))
                step = max(1, len(remaining) // num_remaining_samples)
                samples = [remaining[i] for i in range(0, len(remaining), step)][:num_remaining_samples]
                selected_images = first_batch + samples
                st.caption(f"Analyzing {len(selected_images)} representative pages from all {len(images)} pages")
            else:
                selected_images = images

            # Upload and prepare images for Gemini
            uploaded_images = [image_to_part(img, i) for i, img in enumerate(selected_images)]

            # Generate content using Gemini
            response = model.generate_content([prompt] + uploaded_images)
            return response.text

    except Exception as e:
        st.error(f"Error generating content with Gemini: {str(e)}")
        return None

if uploaded_file is not None:
    # Display the uploaded file
    file_type = uploaded_file.type
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Brochure")
        
        if "pdf" in file_type:
            # Read the bytes once and cache the extraction
            pdf_bytes = uploaded_file.read()
            images = cached_image_extraction(pdf_bytes)
            
            if images:
                st.image(images[0], caption="First page preview", use_column_width=True)
                st.caption(f"PDF with {len(images)} total pages")
            else:
                st.error("Could not extract images from PDF")
        else:  # Image file
            image = process_image(uploaded_file)
            if image:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                images = [image]
            else:
                images = []
    
    with col2:
        st.subheader("Property USPs")
        
        analyze_button = st.button("Extract USPs")
        
        # For better UX, show a placeholder immediately
        result_placeholder = st.empty()
        
        if analyze_button:
            if not setup_gemini_api():
                st.stop()
                
            if not images:
                st.error("No valid images to analyze.")
                st.stop()
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Show thinking state to user
            result_placeholder.info("Thinking... This may take 30-60 seconds depending on the file size.")
            
            # Analyze all images as one brochure
            analysis = analyze_whole_brochure(images, prompt)
            
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
