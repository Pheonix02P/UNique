import streamlit as st
import os
import tempfile
from google import genai
from google.genai import types
import time
import requests
import json
import zipfile
import io
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Premium Property USP Analyzer", layout="wide")
st.title("USP using Gemini")

# Initialize API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Key not found")
    st.stop()

# ── Prompts ────────────────────────────────────────────────────────────────────

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

old_usps_prompt = """
Additionally, I'm providing you with a list of previously identified USPs for this or a similar property. Review these old USPs alongside the brochure.

OLD USPs:
{old_usps}

Merge insights from both sources: remove duplicates, keep the most compelling and unique points from each. Apply the same formatting, character limit, and quality rules to all final USPs.
"""

specifications_prompt = """You are a real estate data extraction specialist. Extract all construction and material specifications from the provided document/source.

═══════════════════════════════════════════
WHAT TO EXTRACT:
═══════════════════════════════════════════
Extract every specification item you can find, including but not limited to:
Structure, Super Structure, Flooring, Plastering, Painting, Electrical, Main Door,
Internal Door, Windows, Kitchen, Toilets / Bathrooms, Water Proofing, Lift, Generator,
CCTV Cameras, Intercom, Security, Plumbing, Terrace, Common Areas, Lobby, Staircase,
External Façade, Compound Wall, Fire Fighting, STP / WTP, Rainwater Harvesting,
Solar, EV Charging, Parking, Gas Pipeline — and any other spec present in the source.

═══════════════════════════════════════════
QUALITY RULES:
═══════════════════════════════════════════
• Extract ONLY data explicitly stated in the source. Do NOT infer or assume.
• Use exact brand names, grades, and measurements as written.
• Preserve all sub-labels (e.g. "Internal:", "External:") within the description.
• If a spec is not mentioned, skip it — do NOT write "Not mentioned" or "N/A".
• The label should be a short, clean heading (e.g. "Structure", "Main Door", "Flooring").
• The description should be a single continuous paragraph capturing ALL detail for that item.
• SENTENCE CASE: Write every description in proper sentence case — first word capitalised, rest lowercase except for brand names, acronyms, and proper nouns (e.g. "Asian Paints", "RCC", "UPVC", "ISI", "BIS", "OHT").

═══════════════════════════════════════════
OUTPUT FORMAT — strict JSON only:
═══════════════════════════════════════════
Return a JSON array where each element is an object with exactly two keys:
  "label"       — short heading for the specification item
  "description" — full detail as a single descriptive paragraph

Example output:
[
  {
    "label": "Structure",
    "description": "Footings, columns, beams and slabs in RCC grade."
  },
  {
    "label": "Flooring",
    "description": "Coral/RAK/Cera/Johnson or equivalent reputed make double charged vitrified tiles with 4\\" skirting."
  },
  {
    "label": "Painting",
    "description": "Internal: Asian/JK/Latif/Equivalent reputed make with Asian easy clean weather coat exterior paints for external walls. Texture finish for elevation. External: Asian Enamel paints over two coats of enamel paint with grey luppum for internal doors and grills."
  },
  {
    "label": "Main Door",
    "description": "Best teak wood door frames and teak wood shutters aesthetically designed with fine finished melamine polishing and designer hardware of reputed make. Height of main door: 7'."
  }
]

Output ONLY the JSON array. No preamble, no explanation, no markdown fences.
"""

PREDEFINED_AMENITIES = [
    "Private Gardens/Balconies", "Swimming Pool", "Internal Street Lights", "Gated Community",
    "Anti-termite Treatment", "Earthquake Resistant", "Paved Compound", "Permeable Pavement",
    "Vastu Compliant", "Wheelchair Accessible", "Grade A Building", "Feng Shui", "Society Office",
    "Heli-Pad", "Solar Lighting", "Well-Maintained Internal Roads", "Energy Efficient Lightining",
    "Community Hall", "Solar Panel", "Temple", "School", "Pet Park", "Solar Water Heating",
    "Co-Working Spaces", "Library", "Carrom", "Thermal Insulation", "Creche/Day Care",
    "Outdoor Event Spaces", "Air Hockey", "Football Ground", "Table Tennis", "Volley Ball Court",
    "Pool Table", "Chess", "Dart Board", "Billiards", "Foosball", "Cricket Pitch", "Bowling Alley",
    "Lawn Tennis Court", "Basketball Court", "Rock Climbing Wall", "Badminton Court",
    "Beach Volley Ball Court", "Spa", "Jacuzzi", "Acupressure Park", "Skating Rink", "Squash Court",
    "Massage Room", "Yoga/Meditation Area", "Sauna", "Futsal", "Reflexology Park", "Aerobics Centre",
    "Video Gaming Room", "Ayurvedic Centre", "Doctor on Call", "Steam Room", "Flower Garden",
    "Terrace Garden", "Medical Centre", "Gymnasium", "Open Space", "Landscape Garden", "Fountain",
    "Clinic", "Pilates Studios", "Natural Pond", "Pedestrian-Friendly Zones", "Manicured Garden",
    "Senior Citizen Sitout", "Archery Range", "Water Park/Slides", "Sit Out Area",
    "Community Garden/Urban Farming", "Green Wall (Vertical Gardens)", "Forest Trail",
    "Cabana Sitting", "Park", "Car Parking", "Art and Craft Studio", "EV Charging Stations",
    "Music Room", "Dance Studio", "Barbecue", "Banquet Hall", "Sun Deck", "Party Lawn", "Sand Pit",
    "Mini Theatre", "Club House", "Children's Play Area", "Multipurpose Hall", "Gazebo",
    "Amphitheatre", "Card Room", "Jogging Track", "Multipurpose court", "Theatre", "Golf Course",
    "Tot Lot", "Nature Trail", "Theater Home", "Cycling Track", "Art Gallery", "Fire Alarm",
    "Gaming Zones", "Boom Barrier", "Wine Cellar", "Emergency Exits", "Golf Simulator",
    "CCTV Camera Security", "Golf Putty", "Fire Fighting Systems", "Security Cabin", "Indoor Games",
    "Gas Leak Detectors", "Biometric/Smart Card Access", "Fire NOC", "Video Door Security",
    "Theme Park", "Smoke Detectors", "24x7 Security", "Panic Buttons in Apartments",
    "Rooftop Lounge", "Car-Free Zones", "Ambulance Service", "Cigar Lounge", "Intercom Facilities",
    "Emergency Evacuation Chairs", "Signage and Road Markings", "Lounge", "Bar/Chill-Out Lounge",
    "Fall Detection Systems in Bathrooms", "Defibrillators in Common Areas", "Piped Gas",
    "Business Lounge", "Restaurant", "Waiting Lounge", "Reading Lounge", "Wi-Fi Connectivity",
    "Pergola", "Smart Home Automation", "DTH Television", "Laundry", "Conference Room",
    "Wi-Fi Zones in Common Areas", "Cafeteria", "RO System", "Food Court", "Laundromat",
    "Shopping Centre", "Property Staff", "Changing Area", "Lifts", "Name Plates",
    "Automated Car Wash", "Concierge Service", "Toilet for Drivers", "Car Wash Area", "Salon",
    "Grocery Shop", "Bus Shelter", "Milk Booth", "Letter Box", "Petrol Pump", "Entrance Lobby",
    "24/7 Power Backup", "Maintenance Staff", "Intercom", "ATM", "DG Availability",
    "Power Back up Lift", "Escalators", "Noise Insulation in Apartments",
    "Centralized Air Conditioning", "Plumber/Electrician on Call", "Secretarial Services",
    "Underground Electric Cabling", "Power Substation", "Braille Signage", "Air Purification Systems",
    "Composting Facilities", "Recycling Facilities", "Garbage Chute", "Garbage Disposal",
    "Organic Waste Converter", "Waste Segregation and Disposal", "Waste Management",
    "Sewage Treatment Plant", "Water Treatment Plant", "Water Softener Plant", "Smart Water Meters",
    "Rain Water Harvesting", "Bioswales", "Ground Water Recharging Systems", "24/7 Water Supply",
    "Municipal Water Supply", "Low Flow Fixtures", "Greywater Recycling", "Borewell Water Supply",
]

amenities_extraction_prompt = """You are an information extraction assistant.

Your task is to extract amenities mentioned in the provided real estate brochure PDF.

IMPORTANT RULES:
1. Only extract amenities that EXACTLY match items from the predefined amenities list provided below.
2. Do NOT add new amenities.
3. Do NOT infer or assume amenities.
4. If a brochure mentions something similar but not exactly matching the list, ignore it.
5. Return the output as a JSON array and nothing else — no preamble, no explanation, no markdown fences.
6. If no amenities from the list are found, return an empty array [].
7. Each amenity should appear only once in the output.
8. Preserve the exact spelling and format from the predefined list.

PREDEFINED AMENITIES LIST:
Private Gardens/Balconies, Swimming Pool, Internal Street Lights, Gated Community,
Anti-termite Treatment, Earthquake Resistant, Paved Compound, Permeable Pavement,
Vastu Compliant, Wheelchair Accessible, Grade A Building, Feng Shui, Society Office,
Heli-Pad, Solar Lighting, Well-Maintained Internal Roads, Energy Efficient Lightining,
Community Hall, Solar Panel, Temple, School, Pet Park, Solar Water Heating,
Co-Working Spaces, Library, Carrom, Thermal Insulation, Creche/Day Care,
Outdoor Event Spaces, Air Hockey, Football Ground, Table Tennis, Volley Ball Court,
Pool Table, Chess, Dart Board, Billiards, Foosball, Cricket Pitch, Bowling Alley,
Lawn Tennis Court, Basketball Court, Rock Climbing Wall, Badminton Court,
Beach Volley Ball Court, Spa, Jacuzzi, Acupressure Park, Skating Rink, Squash Court,
Massage Room, Yoga/Meditation Area, Sauna, Futsal, Reflexology Park, Aerobics Centre,
Video Gaming Room, Ayurvedic Centre, Doctor on Call, Steam Room, Flower Garden,
Terrace Garden, Medical Centre, Gymnasium, Open Space, Landscape Garden, Fountain,
Clinic, Pilates Studios, Natural Pond, Pedestrian-Friendly Zones, Manicured Garden,
Senior Citizen Sitout, Archery Range, Water Park/Slides, Sit Out Area,
Community Garden/Urban Farming, Green Wall (Vertical Gardens), Forest Trail,
Cabana Sitting, Park, Car Parking, Art and Craft Studio, EV Charging Stations,
Music Room, Dance Studio, Barbecue, Banquet Hall, Sun Deck, Party Lawn, Sand Pit,
Mini Theatre, Club House, Children's Play Area, Multipurpose Hall, Gazebo,
Amphitheatre, Card Room, Jogging Track, Multipurpose court, Theatre, Golf Course,
Tot Lot, Nature Trail, Theater Home, Cycling Track, Art Gallery, Fire Alarm,
Gaming Zones, Boom Barrier, Wine Cellar, Emergency Exits, Golf Simulator,
CCTV Camera Security, Golf Putty, Fire Fighting Systems, Security Cabin, Indoor Games,
Gas Leak Detectors, Biometric/Smart Card Access, Fire NOC, Video Door Security,
Theme Park, Smoke Detectors, 24x7 Security, Panic Buttons in Apartments,
Rooftop Lounge, Car-Free Zones, Ambulance Service, Cigar Lounge, Intercom Facilities,
Emergency Evacuation Chairs, Signage and Road Markings, Lounge, Bar/Chill-Out Lounge,
Fall Detection Systems in Bathrooms, Defibrillators in Common Areas, Piped Gas,
Business Lounge, Restaurant, Waiting Lounge, Reading Lounge, Wi-Fi Connectivity,
Pergola, Smart Home Automation, DTH Television, Laundry, Conference Room,
Wi-Fi Zones in Common Areas, Cafeteria, RO System, Food Court, Laundromat,
Shopping Centre, Property Staff, Changing Area, Lifts, Name Plates,
Automated Car Wash, Concierge Service, Toilet for Drivers, Car Wash Area, Salon,
Grocery Shop, Bus Shelter, Milk Booth, Letter Box, Petrol Pump, Entrance Lobby,
24/7 Power Backup, Maintenance Staff, Intercom, ATM, DG Availability,
Power Back up Lift, Escalators, Noise Insulation in Apartments,
Centralized Air Conditioning, Plumber/Electrician on Call, Secretarial Services,
Underground Electric Cabling, Power Substation, Braille Signage, Air Purification Systems,
Composting Facilities, Recycling Facilities, Garbage Chute, Garbage Disposal,
Organic Waste Converter, Waste Segregation and Disposal, Waste Management,
Sewage Treatment Plant, Water Treatment Plant, Water Softener Plant, Smart Water Meters,
Rain Water Harvesting, Bioswales, Ground Water Recharging Systems, 24/7 Water Supply,
Municipal Water Supply, Low Flow Fixtures, Greywater Recycling, Borewell Water Supply

Output ONLY a valid JSON array. No preamble, no explanation, no markdown code fences.
Example: ["Swimming Pool", "Gymnasium", "Club House"]
"""

# ── Helper functions ───────────────────────────────────────────────────────────

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


def analyze_website_via_gemini(url, prompt, model_name, client):
    """Pass the website URL directly to Gemini — no scraping needed."""
    try:
        with st.spinner(f"Analyzing website with {model_name}..."):
            result = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(file_uri=url, mime_type="text/html"),
                    types.Part.from_text(text=prompt),
                ],
            )
        return result.text
    except Exception as e:
        st.error(f"Error during Gemini website analysis: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def analyze_pdf_via_files_api(pdf_bytes, prompt, model_name, client):
    """Upload PDF to Gemini Files API, generate content, then delete the file."""
    uploaded_gemini_file = None
    tmp_path = None

    try:
        with st.spinner("Uploading PDF to Gemini..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            uploaded_gemini_file = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(mime_type="application/pdf"),
            )

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
        if uploaded_gemini_file is not None:
            try:
                client.files.delete(name=uploaded_gemini_file.name)
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


import re

# Words/patterns that must stay uppercase regardless of position
_PRESERVE_UPPER = {
    "RCC", "PVC", "UPVC", "GI", "MS", "SS", "ISI", "BIS", "OHT", "STP", "WTP",
    "ETP", "MLD", "KVA", "KW", "AC", "DC", "MCB", "ELCB", "RCCB", "DB",
    "CCTV", "DTH", "RO", "UV", "TV", "LED", "CFL", "AHU", "VRF", "VRV",
    "RERA", "OC", "CC", "NOC", "BHK", "FSI", "FAR", "TDR",
    "CP", "WPC", "HDF", "MDF", "BWR", "BWP",
}

def _sentence_case(text: str) -> str:
    """
    Convert text to sentence case while preserving:
    - Known uppercase acronyms/abbreviations
    - Brand names (words that start with a capital followed by lowercase)
    - Sub-labels like "Internal:", "External:" (capitalised at sentence start)
    - Measurements, grades, and slash-separated alternatives (e.g. RAK/Cera/Johnson)
    """
    if not text:
        return text

    def fix_sentence(sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return sentence
        words = sentence.split()
        result = []
        for i, word in enumerate(words):
            # Strip trailing punctuation for checking, restore after
            stripped = word.rstrip(".,;:\"')")
            leading = word[: len(word) - len(word.lstrip("(\"'"))]
            trailing = word[len(leading) + len(stripped):]

            # Handle slash-separated tokens (e.g. RAK/Cera/Johnson)
            if "/" in stripped:
                parts = stripped.split("/")
                fixed_parts = []
                for p in parts:
                    p_upper = p.upper()
                    if p_upper in _PRESERVE_UPPER:
                        fixed_parts.append(p_upper)
                    elif p and p[0].isupper():
                        # Treat as brand name — keep capitalisation
                        fixed_parts.append(p)
                    else:
                        fixed_parts.append(p.lower())
                result.append(leading + "/".join(fixed_parts) + trailing)
                continue

            upper_stripped = stripped.upper()
            if upper_stripped in _PRESERVE_UPPER:
                result.append(leading + upper_stripped + trailing)
            elif i == 0:
                # First word of sentence: capitalise first letter only
                result.append(leading + stripped[:1].upper() + stripped[1:].lower() + trailing)
            elif stripped and stripped[0].isupper() and len(stripped) > 1 and stripped[1].islower():
                # Looks like a brand name — preserve as-is
                result.append(word)
            else:
                result.append(leading + stripped.lower() + trailing)

        return " ".join(result)

    # Split on sentence boundaries (. ! ?) keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return "  ".join(fix_sentence(s) for s in sentences)


def apply_sentence_case(specs_data: list) -> list:
    """Apply sentence case to all description fields in the specs list."""
    result = []
    for item in specs_data:
        result.append({
            "label": item.get("label", "").strip(),
            "description": _sentence_case(item.get("description", "").strip()),
        })
    return result


def extract_images_from_pdf(pdf_bytes, min_width=200, min_height=200):
    """
    Extract images from a PDF using PyMuPDF.
    Returns a list of dicts: {name, data (bytes), ext, width, height, page}
    Filters out tiny images (icons, bullets) using min_width/min_height.
    """
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        for img_index, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                width = base_image["width"]
                height = base_image["height"]

                # Skip tiny images (icons, watermarks, decorative elements)
                if width < min_width or height < min_height:
                    continue

                ext = base_image["ext"]
                img_bytes = base_image["image"]

                # Convert CMYK or other color spaces to RGB via pixmap
                if base_image.get("colorspace", 0) not in (1, 3):  # not Gray or RGB
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    ext = "png"

                images.append({
                    "name": f"page{page_num + 1}_img{img_index + 1}.{ext}",
                    "data": img_bytes,
                    "ext": ext,
                    "width": width,
                    "height": height,
                    "page": page_num + 1,
                })
            except Exception:
                continue

    doc.close()
    # Deduplicate by xref — same image embedded on multiple pages appears once
    seen = set()
    unique = []
    for img in images:
        key = (img["width"], img["height"], img["data"][:64])
        if key not in seen:
            seen.add(key)
            unique.append(img)

    return unique



    """Render a flat list of {label, description} spec items."""
    if not specs_data:
        st.info("No specifications could be extracted.")
        return

    for item in specs_data:
        label = item.get("label", "").strip()
        description = item.get("description", "").strip()
        if label and description:
            st.markdown(f"**{label}**")
            st.write(description)
            st.divider()



# ── UI ─────────────────────────────────────────────────────────────────────────

st.write("Upload Brochure, Enter Brochure URL, or Enter Project Website Link — then optionally add Old USPs")

# Model selection
st.subheader("Select Gemini Model")
model_options = {
    "Gemini 3.1 flash": "gemini-3-flash-preview",
    "Gemini 2.5 flash": "gemini-2.5-flash",
}
selected_model_name = st.selectbox(
    "Choose the AI model for analysis / Switch models if facing errors",
    options=list(model_options.keys()),
    index=0,
)
selected_model = model_options[selected_model_name]

# ── Input sources ──────────────────────────────────────────────────────────────
st.subheader("Input Source")

col_upload, col_pdfurl, col_weburl = st.columns(3)

with col_upload:
    st.markdown("**📄 Upload Brochure PDF**")
    uploaded_file = st.file_uploader("Upload a brochure file", type=["pdf"], label_visibility="collapsed")

with col_pdfurl:
    st.markdown("**🔗 Brochure PDF URL**")
    pdf_url = st.text_input(
        "Enter URL to PDF brochure",
        placeholder="https://example.com/brochure.pdf",
        label_visibility="collapsed",
    )

with col_weburl:
    st.markdown("**🌐 Project Website URL**")
    website_url = st.text_input(
        "Enter project website link",
        placeholder="https://projectname.com",
        label_visibility="collapsed",
    )

# Old USPs
st.subheader("Enter Old USPs (Optional)")
old_usps = st.text_area("Paste previous USPs here", height=200)

# ── Resolve input source ───────────────────────────────────────────────────────

pdf_bytes = None
input_source = None
input_mode = None  # "pdf" or "website"

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    input_source = f"File: {uploaded_file.name}"
    input_mode = "pdf"
elif pdf_url and pdf_url.strip():
    pdf_bytes = download_pdf_from_url(pdf_url.strip())
    if pdf_bytes:
        input_source = f"PDF URL: {pdf_url}"
        input_mode = "pdf"
elif website_url and website_url.strip():
    input_source = f"Website: {website_url}"
    input_mode = "website"

# ── Main panels ────────────────────────────────────────────────────────────────

if input_mode:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input Source")
        st.write(input_source)
        if input_mode == "pdf" and pdf_bytes:
            st.info(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")
        elif input_mode == "website":
            st.info("Website content will be scraped on extraction.")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab_usp, tab_amenities, tab_specs, tab_images = st.tabs([
        "📋 Extract USPs",
        "🏊 Extract Amenities",
        "📐 Extract Specifications",
        "🖼️ Extract Images",
    ])

    # ── USP Tab ────────────────────────────────────────────────────────────────
    with tab_usp:
        st.info(f"Model: {selected_model_name}")
        analyze_button = st.button("Extract USPs", key="btn_usp")
        result_placeholder = st.empty()

        if analyze_button:
            client = setup_gemini_client()
            if not client:
                st.stop()

            start_time = time.time()
            result_placeholder.info("Analyzing… this may take 30–90 seconds.")
            analysis = None

            if input_mode == "pdf":
                full_prompt = base_prompt
                if old_usps.strip():
                    full_prompt += old_usps_prompt.format(old_usps=old_usps)
                analysis = analyze_pdf_via_files_api(pdf_bytes, full_prompt, selected_model, client)

            elif input_mode == "website":
                full_prompt = base_prompt
                if old_usps.strip():
                    full_prompt += old_usps_prompt.format(old_usps=old_usps)
                analysis = analyze_website_via_gemini(website_url.strip(), full_prompt, selected_model, client)

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
                    + ("PDF was uploaded to Gemini and deleted after processing." if input_mode == "pdf" else "Website content was analyzed.")
                )

                st.download_button(
                    label="Download USPs",
                    data=analysis,
                    file_name="property_usps.txt",
                    mime="text/plain",
                )
            else:
                result_placeholder.error("Failed to generate analysis. Please try again.")

    # ── Amenities Tab ──────────────────────────────────────────────────────────
    with tab_amenities:
        st.info(f"Model: {selected_model_name}")
        st.write(
            "Extracts only amenities that exactly match the predefined list "
            f"({len(PREDEFINED_AMENITIES)} items)."
        )
        amenities_button = st.button("Extract Amenities", key="btn_amenities")
        amenities_placeholder = st.empty()

        if amenities_button:
            client = setup_gemini_client()
            if not client:
                st.stop()

            start_time = time.time()
            amenities_placeholder.info("Extracting amenities… this may take 30–90 seconds.")
            raw = None

            if input_mode == "pdf":
                raw = analyze_pdf_via_files_api(
                    pdf_bytes, amenities_extraction_prompt, selected_model, client
                )

            elif input_mode == "website":
                raw = analyze_website_via_gemini(website_url.strip(), amenities_extraction_prompt, selected_model, client)

            execution_time = time.time() - start_time

            if raw:
                amenities_placeholder.empty()

                try:
                    clean = raw.strip().strip("```json").strip("```").strip()
                    extracted = json.loads(clean)
                except json.JSONDecodeError:
                    st.error("Model returned unexpected format. Raw response shown below.")
                    st.code(raw)
                    extracted = []

                if extracted:
                    valid = [a for a in extracted if a in PREDEFINED_AMENITIES]
                    invalid = [a for a in extracted if a not in PREDEFINED_AMENITIES]

                    st.subheader(f"Amenities Found ({len(valid)})")
                    cols = st.columns(3)
                    for i, amenity in enumerate(valid):
                        with cols[i % 3]:
                            st.markdown(f"✅ {amenity}")

                    if invalid:
                        st.warning(
                            f"⚠️ {len(invalid)} item(s) returned by the model did not match "
                            "the predefined list and were excluded:"
                        )
                        for item in invalid:
                            st.caption(f"  • {item}")

                    st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")

                    st.download_button(
                        label="Download Amenities (JSON)",
                        data=json.dumps(valid, indent=2),
                        file_name="property_amenities.json",
                        mime="application/json",
                    )
                    st.download_button(
                        label="Download Amenities (TXT)",
                        data="\n".join(valid),
                        file_name="property_amenities.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("No matching amenities were found.")
                    st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")
            else:
                amenities_placeholder.error("Failed to extract amenities. Please try again.")

    # ── Specifications Tab ─────────────────────────────────────────────────────
    with tab_specs:
        st.info(f"Model: {selected_model_name}")
        st.write(
            "Extracts structured project specs: configuration details, construction materials, "
            "common area info, and more."
        )
        specs_button = st.button("Extract Specifications", key="btn_specs")
        specs_placeholder = st.empty()

        if specs_button:
            client = setup_gemini_client()
            if not client:
                st.stop()

            start_time = time.time()
            specs_placeholder.info("Extracting specifications… this may take 30–90 seconds.")
            raw = None

            if input_mode == "pdf":
                raw = analyze_pdf_via_files_api(
                    pdf_bytes, specifications_prompt, selected_model, client
                )

            elif input_mode == "website":
                raw = analyze_website_via_gemini(website_url.strip(), specifications_prompt, selected_model, client)

            execution_time = time.time() - start_time

            if raw:
                specs_placeholder.empty()

                try:
                    clean = raw.strip().strip("```json").strip("```").strip()
                    specs_data = json.loads(clean)
                except json.JSONDecodeError:
                    st.error("Model returned unexpected format. Raw response shown below.")
                    st.code(raw)
                    specs_data = None

                if specs_data:
                    specs_data = apply_sentence_case(specs_data)
                    st.subheader(f"Specifications ({len(specs_data)} items)")
                    render_specifications(specs_data)

                    st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")

                    # Plain text version: "Label\nDescription\n\n"
                    txt_output = "\n\n".join(
                        f"{item['label']}\n{item['description']}"
                        for item in specs_data
                        if item.get("label") and item.get("description")
                    )

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="Download Specifications (JSON)",
                            data=json.dumps(specs_data, indent=2),
                            file_name="property_specifications.json",
                            mime="application/json",
                        )
                    with col_dl2:
                        st.download_button(
                            label="Download Specifications (TXT)",
                            data=txt_output,
                            file_name="property_specifications.txt",
                            mime="text/plain",
                        )
                else:
                    st.info("No specifications could be extracted.")
                    st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")
            else:
                specs_placeholder.error("Failed to extract specifications. Please try again.")

    # ── Images Tab ─────────────────────────────────────────────────────────────
    with tab_images:
        if input_mode == "website":
            st.warning("Image extraction is only available for PDF input (uploaded file or PDF URL). Please provide a brochure PDF.")
        elif not PYMUPDF_AVAILABLE:
            st.error("PyMuPDF is not installed. Add `pymupdf` to your requirements.txt to enable image extraction.")
        else:
            st.write("Extracts all images from the brochure PDF. Tiny icons and decorative elements are filtered out automatically.")

            col_minw, col_minh = st.columns(2)
            with col_minw:
                min_width = st.slider("Minimum image width (px)", 50, 500, 200, step=50)
            with col_minh:
                min_height = st.slider("Minimum image height (px)", 50, 500, 200, step=50)

            images_button = st.button("Extract Images", key="btn_images")
            images_placeholder = st.empty()

            if images_button:
                start_time = time.time()
                images_placeholder.info("Extracting images from PDF…")

                try:
                    extracted_images = extract_images_from_pdf(pdf_bytes, min_width=min_width, min_height=min_height)
                    execution_time = time.time() - start_time
                    images_placeholder.empty()

                    if extracted_images:
                        st.subheader(f"Images Found ({len(extracted_images)})")

                        # Display in a 3-column grid
                        cols = st.columns(3)
                        for i, img in enumerate(extracted_images):
                            with cols[i % 3]:
                                mime = "image/png" if img["ext"] == "png" else f"image/{img['ext']}"
                                st.image(
                                    img["data"],
                                    caption=f"Page {img['page']} — {img['width']}×{img['height']}px",
                                    use_container_width=True,
                                )
                                st.download_button(
                                    label="⬇ Download",
                                    data=img["data"],
                                    file_name=img["name"],
                                    mime=mime,
                                    key=f"dl_img_{i}",
                                )

                        st.divider()

                        # Zip download for all images
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for img in extracted_images:
                                zf.writestr(img["name"], img["data"])
                        zip_buffer.seek(0)

                        st.download_button(
                            label=f"⬇ Download All Images as ZIP ({len(extracted_images)} files)",
                            data=zip_buffer,
                            file_name="brochure_images.zip",
                            mime="application/zip",
                        )

                        st.caption(f"Extraction completed in {execution_time:.1f}s.")
                    else:
                        images_placeholder.info(
                            f"No images found meeting the minimum size ({min_width}×{min_height}px). "
                            "Try lowering the size thresholds."
                        )
                except Exception as e:
                    images_placeholder.error(f"Error extracting images: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

# Footer
st.divider()
st.caption("Premium Property USP Analyzer — Powered by Google Gemini")
