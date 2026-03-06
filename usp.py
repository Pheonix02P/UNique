import streamlit as st
import os
import tempfile
from google import genai
from google.genai import types
import time
import io
import requests

# Page configuration
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Key not found")
    st.stop()

# Set up base prompt
base_prompt = """You are an expert real estate copywriter specializing in premium residential properties. You have been provided a brochure PDF for a premium residential project.

Your task is to extract powerful, factual Unique Selling Propositions (USPs) that will compel high-net-worth buyers to take action.

═══════════════════════════════════════════
EXTRACTION SCOPE — cover ALL of these angles:
═══════════════════════════════════════════
1. Thematic & Architectural Identity — design philosophy, style, signature elements
2. Clubhouse & Lifestyle Amenities — name every facility with its size/count if stated
3. Technology, Automation & Security — smart home, surveillance, access control
4. Landscape, Green & Open Spaces — area %, acres, named gardens, water bodies
5. Location & Connectivity — distances to landmarks, roads, transit hubs (use exact km/min)
6. Developer, Architect & Consultant Pedigree — ONLY if a proper name is explicitly written
7. Awards, Certifications & Approvals — include certifying body and year if available
8. Unit & Project Specifications — total units, density, floors, BHK range, super area
9. Any other distinctive lifestyle, wellness, or convenience feature

═══════════════════════════════════════════
QUALITY RULES (non-negotiable):
═══════════════════════════════════════════
• FACTUAL PRECISION: Every USP must reflect data explicitly present in the brochure.
  - Prefer numbers over adjectives: "2-acre landscaped podium" beats "large garden"
  - Prefer named entities: "Palladian architecture" beats "unique design"
• PROPER NOUNS ONLY: Include architect/designer/consultant names ONLY if a proper name
  (not a generic title) is printed in the brochure.
• NO NOISE: Ignore boilerplate marketing copy, generic slogans, and legal disclaimers.
• NO HEADERS: Output is a flat list — no section titles or groupings.
• GRAMMAR: Capitalize proper nouns; use active, professional language.
• RANKING: Most unique and buyer-influential USP first; descending order throughout.

═══════════════════════════════════════════
CHARACTER LIMIT — STRICTLY ENFORCED:
═══════════════════════════════════════════
• Each USP text must be ≤ 75 characters (including spaces and punctuation).
• Count carefully. If a draft exceeds 75 characters:
    Step 1 — Remove filler words ("featuring", "offering", "boasting").
    Step 2 — Use numerals instead of words ("3" not "three").
    Step 3 — Use "&" instead of "and"; abbreviate units ("sq ft", "km", "min").
    Step 4 — Cut the least important qualifier.
  Never sacrifice a key fact to meet the limit — restructure instead.
• Do NOT truncate mid-word or mid-fact.

═══════════════════════════════════════════
CATEGORIZATION — follow exactly:
═══════════════════════════════════════════
Assign exactly ONE category from this fixed list:
  AMENITIES, LOCATION_AND_CONNECTIVITY, CONSTRUCTION_AND_DESIGN,
  TECHNOLOGY_AND_AUTOMATION, OFFERS, CERTIFICATES_AND_APPROVALS,
  AWARDS_AND_ACCOLADES, MASTER_PLAN

Sub-category rules:

▸ AMENITIES → pick exactly one from this fixed list only:
  ROOFTOP_LOUNGE, CHESS, VISITORS_PARKING, SAUNA, MULTIPURPOSE_COURT,
  ARTS_AND_CRAFTS_STUDIO, MUSIC_ROOM, THEME_PARK, AYURVEDIC_CENTRE,
  MASSAGE_ROOM, SALON, AIR_HOCKEY, GYMNASIUM, FOOSBALL, RESTAURANT,
  SWIMMING_POOL, CAFETERIA, LIBRARY, CARD_ROOM, CO_WORKING_SPACES,
  COMMUNITY_GARDEN_URBAN_FARMING, CLUB_HOUSE, BUSINESS_LOUNGE,
  CRICKET_PITCH, STEAM_ROOM, TOT_LOT, AMPHITHEATRE, ESCALATOR,
  REFLEXOLOGY_PARK, JOGGING_TRACK, CARROM, GREEN_WALL, WATER_PARK_SLIDES,
  INDOOR_GAMES, TABLE_TENNIS, FOOTBALL, SCHOOL, YOGA_MEDITATION_AREA,
  FOOD_COURT, BADMINTON_COURT, MEDICAL_CENTRE, CIGAR_LOUNGE, CLINIC,
  FLOWER_GARDEN, SQUASH_COURT, BILLIARDS, CAR_WASH_AREA, GAZEBO, PARKING,
  LANDSCAPE_GARDEN, TEMPLE, BARBEQUE, CYCLING_TRACK, CRECHE, LIFT,
  THEATER_HOME, SENIOR_CITIZEN_SITOUT, AEROBICS_CENTRE, AUTOMATED_CAR_WASH,
  BANQUET_HALL, SAND_PIT, PEDESTRIAN_FRIENDLY_ZONES, MULTIPURPOSE_HALL,
  EXTERIOR_LANDSCAPE, CAR_PARKING, GAMING_ZONES, PRIVATE_GARDENS_BALCONIES,
  MINI_THEATRE, GROCERY_SHOP, TERRACE_GARDEN, ARCHERY_RANGE, GOLF_COURSE,
  ATM, SKATING_RINK, BASKETBALL_COURT, NATURE_TRAIL, SHOPPING_CENTRE,
  PERGOLA, POOL_TABLE, PAVED_COMPOUND, LOUNGE, TODDLER_POOL,
  COMMUNITY_HALL, PARTY_LAWN, READING_LOUNGE, FOUNTAIN, JACUZZI,
  POWER_SUBSTATION, CENTRALIZED_AIR_CONDITIONING, SIT_OUT_AREA,
  CHILDRENS_PLAY_AREA, LAWN_TENNIS_COURT, SPA, BAR_CHILL_OUT_LOUNGE,
  INTERNAL_ROAD, THEATRE, BOWLING_ALLEY, MANICURED_GARDEN,
  ACUPRESSURE_PARK, CONFERENCE_ROOM, FOREST_TRAIL,
  BEACH_VOLLEY_BALL_COURT, INFINITY_POOL, ACCUPRESSURE_PARK, OPEN_SPACE,
  DANCE_STUDIO, SUN_DECK, NATURAL_POND, ROCK_CLIMBING_WALL, DART_BOARD,
  EV_CHARGING_STATIONS
  → If no suitable sub-category exists, do NOT use AMENITIES; do NOT create a custom one.

▸ LOCATION_AND_CONNECTIVITY → pick exactly one from this fixed list only:
  BUS, ELECTRICITY, BEACH, PETROL_PUMP, COLLEGE, AIRPORT, MARKETS, METRO,
  STADIUM, HOSPITALITY, GREEN_BELT, PARK, WATER, GOLF_COURSE, ATM,
  HERITAGE_PLACES, BANK, MULTI_LEVEL_PARKING, RAILWAY, AMUSEMENT_PARK,
  HIGHWAY, HOSPITAL, FLYOVER, MALLS, SCHOOL, MAJOR_ROAD, BUSINESS_HUB,
  PUBLIC_TRANSPORTATION
  → If no suitable sub-category exists, do NOT use LOCATION_AND_CONNECTIVITY; do NOT create a custom one.

▸ All other categories → create a custom sub-category in Title Case.
  It must NOT duplicate any predefined category or sub-category name.

═══════════════════════════════════════════
OUTPUT FORMAT — one line per USP, nothing else:
═══════════════════════════════════════════
[CATEGORY] | [SUB_CATEGORY] | [USP text ≤ 75 characters]

Examples of correct output:
AMENITIES | CLUB_HOUSE | 1 lakh sq ft clubhouse — largest in the micro-market
AMENITIES | SWIMMING_POOL | Infinity pool at 40th floor with panoramic city views
CONSTRUCTION_AND_DESIGN | Palladian Architecture | Neo-classical façade by award-winning Hafeez Contractor
LOCATION_AND_CONNECTIVITY | METRO | 5-min walk to Magenta Line metro station
MASTER_PLAN | Low Density | Only 400 units across 10 acres — 40 units/acre density
TECHNOLOGY_AND_AUTOMATION | Smart Home | Alexa-enabled smart home automation in every unit
CERTIFICATES_AND_APPROVALS | RERA Status | RERA registered; OC received — ready to move in
AWARDS_AND_ACCOLADES | Cnbc Awaaz Award | CNBC Awaaz Real Estate Award 2023 — Best Luxury Project

Begin extraction now. Output ONLY the formatted USP lines. No preamble, no summary, no extra text.
"""

# Additional prompt for when old USPs are provided
old_usps_prompt = """
Additionally, I'm providing you with a list of previously identified USPs for this or a similar property. Review these old USPs alongside the brochure.

OLD USPs:
{old_usps}

Merge insights from both sources: remove duplicates, keep the most compelling and unique points from each. Apply the same formatting, character limit, and quality rules to all final USPs.
"""

# Main content area
st.write("Upload Brochure or Enter URL and (Optionally) Enter Old USPs")

# Model selection
st.subheader("Select Gemini Model")
model_options = {
    "Gemini 3.1 Pro Preview": "gemini-3.1-pro-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro-preview-06-05",
}
selected_model_name = st.selectbox(
    "Choose the AI model for analysis / Switch models if facing errors",
    options=list(model_options.keys()),
    index=0,
)
selected_model = model_options[selected_model_name]

# File uploader
uploaded_file = st.file_uploader("Upload a brochure file", type=["pdf"])

# URL input
st.write("OR")
pdf_url = st.text_input("Enter URL to PDF brochure", placeholder="https://example.com/brochure.pdf")

# Old USPs
st.subheader("Enter Old USPs (Optional)")
old_usps = st.text_area("Paste previous USPs here", height=200)


def setup_gemini_client():
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None


def download_pdf_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
            st.error(f"URL does not point to a valid PDF. Content-Type: {content_type}")
            return None
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None


def analyze_pdf_via_files_api(pdf_bytes, prompt, model_name, client):
    """Upload PDF to Gemini Files API, generate content, then delete the file."""
    uploaded_gemini_file = None
    tmp_path = None

    try:
        with st.spinner(f"Uploading PDF to Gemini..."):
            # Write bytes to a temp file so the SDK can upload it
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            uploaded_gemini_file = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(mime_type="application/pdf"),
            )

            # Poll until the file is ACTIVE
            max_wait = 60
            waited = 0
            while uploaded_gemini_file.state.name != "ACTIVE":
                if waited >= max_wait:
                    st.error("Gemini file processing timed out.")
                    return None
                time.sleep(3)
                waited += 3
                uploaded_gemini_file = client.files.get(name=uploaded_gemini_file.name)

        with st.spinner(f"Analyzing with {model_name}..."):
            result = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_gemini_file.uri,
                        mime_type="application/pdf",
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )

        return result.text

    except Exception as e:
        st.error(f"Error during Gemini analysis: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

    finally:
        # Always delete the uploaded file from Gemini and the local temp file
        if uploaded_gemini_file is not None:
            try:
                client.files.delete(name=uploaded_gemini_file.name)
            except Exception:
                pass  # Best-effort cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── Main flow ──────────────────────────────────────────────────────────────────

pdf_bytes = None
input_source = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
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
        st.info(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")

    with col2:
        st.subheader("Property USPs")
        st.info(f"Model: {selected_model_name}")
        analyze_button = st.button("Extract USPs")
        result_placeholder = st.empty()

        if analyze_button:
            client = setup_gemini_client()
            if not client:
                st.stop()

            start_time = time.time()
            result_placeholder.info("Uploading PDF and analyzing… this may take 30–90 seconds.")

            full_prompt = base_prompt
            if old_usps.strip():
                full_prompt += old_usps_prompt.format(old_usps=old_usps)

            analysis = analyze_pdf_via_files_api(pdf_bytes, full_prompt, selected_model, client)
            execution_time = time.time() - start_time

            if analysis:
                result_placeholder.empty()

                usps = [u for u in analysis.strip().split("\n") if u.strip()]

                st.subheader("Extracted USPs")

                for usp in usps:
                    parts = usp.split("|")
                    if len(parts) == 3:
                        category = parts[0].strip()
                        subcategory = parts[1].strip()
                        text = parts[2].strip()
                        char_count = len(text)
                        over = char_count > 75

                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.markdown(f"**{category}**")
                            st.caption(subcategory)
                        with col_b:
                            st.markdown(f"• {text}")
                            if over:
                                st.warning(f"⚠️ {char_count} chars (limit: 75)")
                        st.divider()
                    else:
                        st.markdown(f"• {usp}")

                st.caption(
                    f"Analysis completed in {execution_time:.1f}s using {selected_model_name}. "
                    f"PDF was uploaded to Gemini and deleted after processing."
                )

                st.download_button(
                    label="Download USPs",
                    data=analysis,
                    file_name="property_usps.txt",
                    mime="text/plain",
                )
            else:
                result_placeholder.error("Failed to generate analysis. Please try again.")

# Footer
st.divider()
st.caption("Premium Property USP Analyzer — Powered by Google Gemini")
