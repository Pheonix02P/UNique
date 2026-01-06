import streamlit as st
import os
import tempfile
from google import genai
import time
import fitz  # PyMuPDF for PDF rendering
import io
from PIL import Image
import requests

# Page configurationg
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
try:
   GEMINI_API_KEY=st.secrets["GEMINI_API_KEY"]
except KeyError:
   st.error("Key not found")
   st.stop()
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

CATEGORIZATION REQUIREMENTS:

For each USP, do the following:

1. Assign a category:
   - Choose exactly one from this fixed list (no additions or changes allowed):
     AMENITIES, LOCATION_AND_CONNECTIVITY, CONSTRUCTION_AND_DESIGN,
     TECHNOLOGY_AND_AUTOMATION, OFFERS, CERTIFICATES_AND_APPROVALS,
     AWARDS_AND_ACCOLADES, MASTER_PLAN.

2. Select a sub-category:

   - If category is **AMENITIES**, pick exactly one sub-category only from this fixed list (no additions or changes allowed):
     ROOFTOP_LOUNGE, CHESS, VISITORS_PARKING, SAUNA, MULTIPURPOSE_COURT, ARTS_AND_CRAFTS_STUDIO, MUSIC_ROOM, THEME_PARK, AYURVEDIC_CENTRE, MASSAGE_ROOM, SALON, AIR_HOCKEY, GYMNASIUM, FOOSBALL, RESTAURANT, SWIMMING_POOL, CAFETERIA, LIBRARY, CARD_ROOM, CO_WORKING_SPACES, COMMUNITY_GARDEN_URBAN_FARMING, CLUB_HOUSE, BUSINESS_LOUNGE, CRICKET_PITCH, STEAM_ROOM, TOT_LOT, AMPHITHEATRE, ESCALATOR, REFLEXOLOGY_PARK, JOGGING_TRACK, CARROM, GREEN_WALL, WATER_PARK_SLIDES, INDOOR_GAMES, TABLE_TENNIS, FOOTBALL, SCHOOL, YOGA_MEDITATION_AREA, FOOD_COURT, BADMINTON_COURT, MEDICAL_CENTRE, CIGAR_LOUNGE, CLINIC, FLOWER_GARDEN, SQUASH_COURT, BILLIARDS, CAR_WASH_AREA, GAZEBO, PARKING, LANDSCAPE_GARDEN, TEMPLE, BARBEQUE, CYCLING_TRACK, CRECHE, LIFT, THEATER_HOME, SENIOR_CITIZEN_SITOUT, AEROBICS_CENTRE, AUTOMATED_CAR_WASH, BANQUET_HALL, SAND_PIT, PEDESTRIAN_FRIENDLY_ZONES, MULTIPURPOSE_HALL, EXTERIOR_LANDSCAPE, CAR_PARKING, GAMING_ZONES, PRIVATE_GARDENS_BALCONIES, MINI_THEATRE, GROCERY_SHOP, TERRACE_GARDEN, ARCHERY_RANGE, GOLF_COURSE, ATM, SKATING_RINK, BASKETBALL_COURT, NATURE_TRAIL, SHOPPING_CENTRE, PERGOLA, POOL_TABLE, PAVED_COMPOUND, LOUNGE, TODDLER_POOL, COMMUNITY_HALL, PARTY_LAWN, READING_LOUNGE, FOUNTAIN, JACUZZI, POWER_SUBSTATION, CENTRALIZED_AIR_CONDITIONING, SIT_OUT_AREA, CHILDRENS_PLAY_AREA, LAWN_TENNIS_COURT, SPA, BAR_CHILL_OUT_LOUNGE, INTERNAL_ROAD, THEATRE, BOWLING_ALLEY, MANICURED_GARDEN, ACUPRESSURE_PARK, CONFERENCE_ROOM, FOREST_TRAIL, BEACH_VOLLEY_BALL_COURT, INFINITY_POOL, ACCUPRESSURE_PARK, OPEN_SPACE, DANCE_STUDIO, SUN_DECK, NATURAL_POND, ROCK_CLIMBING_WALL, DART_BOARD, EV_CHARGING_STATIONS.
     
     - If no suitable sub-category is found, do **not** assign AMENITIES as the category and do **not** generate a custom subcategory.

   - If category is **LOCATION_AND_CONNECTIVITY**, pick exactly one sub-category only from this fixed list (no additions or changes allowed):
     BUS, ELECTRICITY, BEACH, PETROL_PUMP, COLLEGE, AIRPORT, MARKETS, METRO, STADIUM, HOSPITALITY, GREEN_BELT, PARK, WATER, GOLF_COURSE, ATM, HERITAGE_PLACES, BANK, MULTI_LEVEL_PARKING, RAILWAY, AMUSEMENT_PARK, HIGHWAY, HOSPITAL, FLYOVER, MALLS, SCHOOL, MAJOR_ROAD, BUSINESS_HUB, PUBLIC_TRANSPORTATION.
     
     - If no suitable sub-category is found, do **not** assign LOCATION_AND_CONNECTIVITY as the category and do **not** generate a custom subcategory.

   - For all other categories:
     - Create a custom sub-category (not from any predefined list), in **Title Case Format**.
     - This custom sub-category must not duplicate any predefined category or sub-category.

OUTPUT FORMAT:

Present each USP in the following format:
[Category] | [Sub-Category] | [USP Text within 75 characters]

Example:
AMENITIES | SWIMMING_POOL | Olympic-size pool with cabanas and kids' splash zone
CONSTRUCTION_AND_DESIGN | Sustainable Materials | LEED Gold certified with eco-friendly construction materials

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

# Model selection dropdown
st.subheader("Select Gemini Model")
model_options = {
    "Gemini": "gemini-3-flash-preview",
}
selected_model_name = st.selectbox(
    "Choose the AI model for analysis/Switch Models while facing any issue or errors",
    options=list(model_options.keys()),
    index=0,
    help="Gemini 2.0 Flash is faster and cost-effective."
)
selected_model = model_options[selected_model_name]

# File uploader
uploaded_file = st.file_uploader("Upload a brochure file", type=["pdf"])

# URL input
st.write("OR")
pdf_url = st.text_input("Enter URL to PDF brochure", placeholder="https://example.com/brochure.pdf")

# Text area for old USPs
st.subheader("Enter Old USPs (Optional)")
old_usps = st.text_area("Paste previous USPs here", height=200)

def setup_gemini_client():
    """Initialize the Gemini client"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None

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
       
def analyze_pdf(pdf_bytes, prompt, model_name, client):
    """Analyze PDF directly with Gemini using the new SDK"""
    try:
        with st.spinner(f"Analyzing brochure with {model_name}..."):
            # Create a temporary file to store the PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(pdf_bytes)
            temp_file.close()
            
            # Upload the PDF file using the new SDK method
            uploaded_file = client.files.upload(file=temp_file.name)
            
            # Wait for the file to be processed and reach ACTIVE state
            st.info(f"File uploaded. Waiting for processing... (File ID: {uploaded_file.name})")
            max_wait_time = 120
            wait_interval = 2
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                file_status = client.files.get(name=uploaded_file.name)
                
                if file_status.state.name == "ACTIVE":
                    st.success("File is ready for processing!")
                    break
                elif file_status.state.name == "FAILED":
                    st.error("File processing failed on Gemini's end.")
                    return None
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                
                if elapsed_time % 10 == 0:
                    st.info(f"Still processing... ({elapsed_time}s elapsed)")
            
            if elapsed_time >= max_wait_time:
                st.error("Timeout: File did not become active within the expected time.")
                return None
            
            # CRITICAL FIX: Use the correct content structure
            # Create a proper message with parts
            result = client.models.generate_content(
                model=model_name,
                contents={
                    "parts": [
                        {"text": prompt},
                        {
                            "file_data": {
                                "mime_type": uploaded_file.mime_type,
                                "file_uri": uploaded_file.uri
                            }
                        }
                    ]
                }
            )
            
            # Clean up the temporary file
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as e:
                pass
            
            # Optionally delete the file from Gemini's servers
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                pass
            
            # Extract text from response
            return result.text

    except Exception as e:
        st.error(f"Error generating content with Gemini: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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
            st.image(preview_image, caption="First page preview", width='content')
        else:
            st.warning("Could not generate preview for this PDF")
    
    with col2:
        st.subheader("Property USPs")
        
        # Display selected model
        st.info(f"Using model: {selected_model_name}")
        
        analyze_button = st.button("Extract USPs")
        
        # For better UX, show a placeholder immediately
        result_placeholder = st.empty()
        
        if analyze_button:
            # Initialize Gemini client
            client = setup_gemini_client()
            if not client:
                st.stop()
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Show thinking state to user
            result_placeholder.info("Thinking... This may take 30-60 seconds depending on the file size.")
            
            # Prepare the prompt based on whether old USPs are provided
            full_prompt = base_prompt
            if old_usps.strip():
                full_prompt += old_usps_prompt.format(old_usps=old_usps)
            
            # Analyze PDF with selected model using new SDK
            analysis = analyze_pdf(pdf_bytes, full_prompt, selected_model, client)
            
            # Display execution time
            execution_time = time.time() - start_time
            
            if analysis:
                # Clear placeholder and show result
                result_placeholder.empty()
                
                # Parse and display USPs in a formatted way
                usps = analysis.strip().split('\n')
                
                st.subheader("Extracted USPs")
                
                # Display each USP in a more readable format
                for i, usp in enumerate(usps, 1):
                    if usp.strip():
                        # Split by pipe character
                        parts = usp.split('|')
                        if len(parts) == 3:
                            category = parts[0].strip()
                            subcategory = parts[1].strip()
                            text = parts[2].strip()
                            
                            # Display with better formatting
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"**{category}**")
                                st.caption(subcategory)
                            with col_b:
                                st.markdown(f"• {text}")
                            st.divider()
                        else:
                            # Fallback for non-formatted lines
                            st.markdown(f"• {usp}")
                
                st.caption(f"Analysis completed in {execution_time:.1f} seconds using {selected_model_name}")
                
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



















