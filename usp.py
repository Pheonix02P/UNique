import streamlit as st
import os
import tempfile
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF for PDF handling
import io
import google.generativeai as genai
import time
import concurrent.futures

# Enhanced caching strategy - cache both extraction and enhancement
@st.cache_data
def cached_image_extraction(pdf_bytes):
    """Cached function to extract images from PDF bytes"""
    return extract_images_from_pdf_bytes(pdf_bytes)

@st.cache_data
def cached_image_enhancement(img_bytes, max_size=2000):
    """Cache the image enhancement process"""
    try:
        image = Image.open(io.BytesIO(img_bytes))
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size/image.width, max_size/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        enhanced = enhance_image_for_text_recognition(image)
        return enhanced
    except Exception as e:
        st.warning(f"Enhancement warning: {str(e)}")
        return Image.open(io.BytesIO(img_bytes))

# Page configuration
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
GEMINI_API_KEY = "AIzaSyCIR8-WfadSCfOZTr1PxJFXRzP5HbiE9IQ"

# The same prompt as before
prompt = """You are provided with the attached brochure for a premium residential project. Your task is to extract the unique
selling propositions (USPs) that will positively influence potential buyer decisions, keeping in mind the expectations of buyers in this segment.
Focus on the following aspects, aligning with the expectations of a premium homebuyer:
-            Thematic and Architectural Uniqueness
-            Facilities and Luxury Amenities
-            Technology and Security Features  
-            Landscape and Environment  
-            Location Highlights  
-            Awards and Recognition
-            Any Other
 
Unique Features that enhance lifestyle, convenience, and security.
NOTE:
-            Keep in mind that the attachment may contain noise, so filter out any irrelevant content.
-            Output the USPs as bullet points, ensuring each bullet point is 20 words or less.
-            Ensure each point provides factual details about the project based on the information
available in the brochure.

*Important Instructions for Proper Names*: 
- Pay SPECIAL ATTENTION to accurately spelling the proper names of architects, designers, builders, consultants, and developers
- Double-check all proper names to ensure they are spelled EXACTLY as they appear in the brochure
- If a proper name appears multiple times with different spellings, use the most frequent spelling
- If and only if the proper name of an architect, designer, builder, consultant, or developer is explicitly mentioned in the brochure, include it in the USPs
- Do not use common nouns such as "designers" or "architect" without the presence of a proper noun

- Arrange USPs in descending order, with the most unique and attractive USP at the top.
- Give priority to factual details explicitly mentioned in the text, such as the size of the clubhouse, project density, and greenery.
- Use a professional tone in your bullet points.
- Do not include headers in the bullet points.
- Ensure grammatical correctness and capitalize the first letters of proper nouns.
- Focus on: (factual information, lifestyle appeal, and renowned names associated with the project).
- Include unique points and factual information from the following reference points given to you.
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
        # Convert to bytes first for caching
        img_bytes = image_file.read()
        return cached_image_enhancement(img_bytes)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def enhance_image_for_text_recognition(image):
    """Optimized image processing - reduce processing intensity"""
    try:
        # Make a copy to avoid modifying original
        enhanced = image.copy()
        
        # Apply more conservative adjustments for speed
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        # Skip brightness adjustment if not necessary
        
        # Apply lighter sharpening for faster processing
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        
        return enhanced
    except Exception as e:
        st.warning(f"Image enhancement warning (continuing with original): {str(e)}")
        return image

def extract_images_from_pdf_bytes(pdf_bytes):
    """Extract all images from PDF with parallel processing for speed"""
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
        
        # Process all pages with optimized parameters
        with st.spinner(f"Extracting all {total_pages} pages..."):
            def process_page(page_num):
                page = doc.load_page(page_num)
                
                # Reduced resolution for faster processing while maintaining quality
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                img_data = pix.tobytes("ppm")
                
                # Convert to bytes for caching enhancement
                img_bytes = io.BytesIO()
                img = Image.open(io.BytesIO(img_data))
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                
                # Use cached enhancement
                return cached_image_enhancement(img_bytes.getvalue(), max_size=2200)
            
            # Use parallel processing for faster image extraction
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Process pages concurrently (limit concurrency to avoid memory issues)
                max_workers = min(8, os.cpu_count() or 4)
                chunk_size = max(1, total_pages // max_workers)
                
                # Process in smaller batches to prevent memory overload
                all_pages = list(range(total_pages))
                for i in range(0, total_pages, chunk_size):
                    chunk = all_pages[i:i+chunk_size]
                    futures = {executor.submit(process_page, page_num): page_num for page_num in chunk}
                    
                    for future in concurrent.futures.as_completed(futures):
                        page_num = futures[future]
                        try:
                            img = future.result()
                            if img:
                                images.append((page_num, img))
                        except Exception as e:
                            st.warning(f"Error processing page {page_num}: {str(e)}")
            
            # Sort images by page number
            images.sort(key=lambda x: x[0])
            images = [img for _, img in images]
        
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

# Optimized text extraction with caching
@st.cache_data
def cached_text_extraction(pdf_bytes):
    """Cache PDF text extraction"""
    return extract_text_directly_from_pdf(pdf_bytes)

def extract_text_directly_from_pdf(pdf_bytes):
    """Extract text from all pages of the PDF"""
    extracted_text = ""
    temp_file = None
    
    try:
        # Create a temporary file for the PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(pdf_bytes)
        temp_file.close()
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(temp_file.name)
        total_pages = len(doc)
        
        # Extract text from all pages
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            extracted_text += f"[Page {page_num+1}]\n" + page.get_text() + "\n\n"
        
        doc.close()
        return extracted_text
    except Exception as e:
        st.warning(f"Text extraction warning: {str(e)}")
        return ""
    finally:
        # Clean up
        try:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            pass

def analyze_whole_brochure(images, prompt, extracted_text=""):
    """Analyze brochure with all pages, but use smart sampling for Gemini API"""
    try:
        with st.spinner("Analyzing brochure with AI..."):
            # Initialize the Gemini model - use flash version for speed
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Convert image to part with optimized quality
            def image_to_part(image, index):
                # Create BytesIO object instead of temporary file
                img_bytes = io.BytesIO()
                # Use optimized quality setting - balance size and quality
                image.save(img_bytes, format="JPEG", quality=85)
                img_bytes.seek(0)
                
                # Upload directly from BytesIO
                return genai.upload_file(img_bytes, mime_type="image/jpeg", display_name=f"page_{index+1}.jpg")

            # For Gemini API, we still need to sample if there are too many pages
            # as the API has token limits
            if len(images) > 8:
                # Always include first 4 pages (often contain key information)
                first_batch = images[:4]
                
                # Include last 2 pages (often contain contact info and important details)
                last_batch = images[-2:] if len(images) > 6 else []
                
                # Sample from middle pages
                remaining = images[4:-2] if len(images) > 6 else images[4:]
                num_remaining_samples = min(6, len(remaining))  # Sample up to 6 more pages
                
                if remaining:
                    step = max(1, len(remaining) // num_remaining_samples)
                    middle_samples = [remaining[i] for i in range(0, len(remaining), step)][:num_remaining_samples]
                else:
                    middle_samples = []
                
                selected_images = first_batch + middle_samples + last_batch
                st.caption(f"Sending {len(selected_images)} representative pages to Gemini API from all {len(images)} extracted pages")
            else:
                selected_images = images

            # Prepare images in parallel for speed
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Process images concurrently
                future_to_idx = {executor.submit(image_to_part, img, i): i for i, img in enumerate(selected_images)}
                uploaded_images = [None] * len(selected_images)
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        uploaded_images[idx] = future.result()
                    except Exception as e:
                        st.warning(f"Error uploading image {idx}: {str(e)}")
            
            # Filter out any None values from failed uploads
            uploaded_images = [img for img in uploaded_images if img is not None]
            
            # Optimize text context - trim to essential parts
            content_parts = [prompt]
            if extracted_text:
                # Limit text to avoid token issues but keep enough context
                max_text_len = 20000
                content_parts.append(f"Text context (focus on proper names):\n{extracted_text[:max_text_len]}")
            
            content_parts.extend(uploaded_images)

            # Optimized generation config for balanced performance
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            response = model.generate_content(
                content_parts,
                generation_config=generation_config,
                stream=False
            )
            
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
        
        extracted_text = ""
        if "pdf" in file_type:
            # Show a progress indicator immediately 
            with st.spinner("Processing PDF..."):
                # Read the bytes once and use for both extraction methods
                pdf_bytes = uploaded_file.read()
                
                # Extract ALL images from the PDF
                images = cached_image_extraction(pdf_bytes)
                
                # Get text from ALL pages in parallel with image extraction
                extracted_text = cached_text_extraction(pdf_bytes)
                
                if images:
                    st.image(images[0], caption="First page preview", use_container_width=True)
                    st.caption(f"PDF processed with all {len(images)} pages extracted")
                else:
                    st.error("Could not extract images from PDF")
        else:  # Image file
            image = process_image(uploaded_file)
            if image:
                st.image(image, caption="Uploaded Image", use_container_width=True)
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
            
            # Show progress immediately
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Starting analysis...")
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Update progress
            progress_bar.progress(40)
            status_text.info("Preparing images for analysis... (40%)")
            
            # Analyze images
            analysis = analyze_whole_brochure(images, prompt, extracted_text)
            
            # Update progress
            progress_bar.progress(90)
            status_text.info("Finalizing results... (90%)")
            
            # Display execution time
            execution_time = time.time() - start_time
            
            if analysis:
                # Clear placeholder and progress indicators
                status_text.empty()
                progress_bar.progress(100)
                
                # Show results
                st.markdown(analysis)
                st.success(f"Analysis completed in {execution_time:.1f} seconds")
                
                # Option to download results
                st.download_button(
                    label="Download USPs",
                    data=analysis,
                    file_name="property_usps.txt",
                    mime="text/plain"
                )
            else:
                status_text.error("Failed to generate analysis. Please try again.")
                progress_bar.empty()

# Footer
st.divider()
st.caption("Premium Property USP Analyzer - Powered by Google Gemini")