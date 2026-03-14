import streamlit as st
import os
import tempfile
from google import genai
from google.genai import types
import time
import requests
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Intel · USP Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(196,160,100,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(120,90,200,0.05) 0%, transparent 60%),
        #0a0a0f !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* Hero */
.hero-wrap {
    position: relative;
    padding: 3.5rem 0 2.5rem;
    margin-bottom: 2rem;
}
.hero-line {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c4a064, #9b7fd4, transparent);
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%,100% { opacity:0.4; transform:scaleX(0.8); }
    50%      { opacity:1;   transform:scaleX(1); }
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #c4a064;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    animation: fadeUp 0.6s ease both;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 700;
    line-height: 1.1;
    color: #f0ece3;
    margin: 0 0 0.5rem;
    animation: fadeUp 0.7s ease both;
}
.hero-title span {
    background: linear-gradient(135deg, #c4a064 30%, #e8c97e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8a8474;
    letter-spacing: 0.02em;
    animation: fadeUp 0.8s ease both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}

/* Cards */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(196,160,100,0.12);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s, box-shadow 0.3s;
    animation: fadeUp 0.5s ease both;
}
.card:hover {
    border-color: rgba(196,160,100,0.28);
    box-shadow: 0 8px 40px rgba(196,160,100,0.07);
}
.card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c4a064;
    margin-bottom: 1rem;
}

/* Streamlit widget reskins */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(196,160,100,0.18) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(196,160,100,0.5) !important;
    box-shadow: 0 0 0 3px rgba(196,160,100,0.08) !important;
    outline: none !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(196,160,100,0.25) !important;
    border-radius: 12px !important;
    background: rgba(196,160,100,0.03) !important;
    padding: 0.5rem !important;
    transition: border-color 0.3s, background 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(196,160,100,0.45) !important;
    background: rgba(196,160,100,0.06) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c4a064, #a8854a) !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(196,160,100,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(196,160,100,0.4) !important;
    opacity: 0.92 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #c4a064 !important;
    border: 1px solid rgba(196,160,100,0.35) !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(196,160,100,0.08) !important;
    border-color: rgba(196,160,100,0.6) !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(196,160,100,0.15) !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #6b6456 !important;
    padding: 0.7rem 1.6rem !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.2s, border-color 0.2s !important;
    letter-spacing: 0.03em;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #c4a064 !important;
    border-bottom-color: #c4a064 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #e8c97e !important; }
[data-testid="stTabContent"] {
    padding-top: 1.5rem !important;
    animation: fadeUp 0.4s ease both;
}

/* Labels */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stFileUploader"] label {
    color: #8a8474 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
}

hr { border-color: rgba(196,160,100,0.1) !important; margin: 1.2rem 0 !important; }

/* USP items */
.usp-item {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.6rem;
    transition: background 0.2s, border-color 0.2s, transform 0.2s;
    animation: fadeUp 0.4s ease both;
}
.usp-item:hover {
    background: rgba(196,160,100,0.06);
    border-color: rgba(196,160,100,0.2);
    transform: translateX(4px);
}
.usp-badge {
    flex-shrink: 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.55rem;
    border-radius: 6px;
    white-space: nowrap;
    margin-top: 2px;
}
.usp-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #6b6456;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}
.usp-text { font-size: 0.9rem; color: #d4cfc5; line-height: 1.45; }
.usp-warn { font-size: 0.72rem; color: #e8834a; margin-top: 0.25rem; }

/* Amenity chips */
.amenity-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    background: rgba(196,160,100,0.08);
    border: 1px solid rgba(196,160,100,0.18);
    color: #d4c9af;
    font-size: 0.8rem;
    margin: 0.25rem;
    transition: background 0.2s, border-color 0.2s, transform 0.15s;
    animation: popIn 0.3s ease both;
    cursor: default;
}
.amenity-chip:hover {
    background: rgba(196,160,100,0.16);
    border-color: rgba(196,160,100,0.38);
    transform: scale(1.04);
}
.amenity-chip .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #c4a064;
    flex-shrink: 0;
}
@keyframes popIn {
    from { opacity:0; transform:scale(0.85); }
    to   { opacity:1; transform:scale(1); }
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 1.5rem;
    padding: 1rem 1.5rem;
    background: rgba(196,160,100,0.05);
    border: 1px solid rgba(196,160,100,0.12);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #c4a064;
    line-height: 1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #6b6456;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Section heading */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    color: #f0ece3;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(196,160,100,0.2), transparent);
}

/* OR divider */
.or-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.8rem 0;
    color: #4a4540;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
}
.or-divider::before, .or-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(196,160,100,0.1);
}

/* PDF pill */
.pdf-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background: rgba(196,160,100,0.08);
    border: 1px solid rgba(196,160,100,0.2);
    font-size: 0.8rem;
    color: #c4a064;
    font-family: 'DM Mono', monospace;
    animation: fadeUp 0.4s ease both;
}

/* Timing note */
.timing-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #5a5248;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}

/* Footer */
.footer-caption {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #3a3530;
    text-align: center;
    padding: 2rem 0 0.5rem;
}

/* Stagger for USP items */
.usp-item:nth-child(1)  { animation-delay:0.05s; }
.usp-item:nth-child(2)  { animation-delay:0.10s; }
.usp-item:nth-child(3)  { animation-delay:0.15s; }
.usp-item:nth-child(4)  { animation-delay:0.20s; }
.usp-item:nth-child(5)  { animation-delay:0.25s; }
.usp-item:nth-child(6)  { animation-delay:0.30s; }
.usp-item:nth-child(7)  { animation-delay:0.35s; }
.usp-item:nth-child(8)  { animation-delay:0.40s; }
.usp-item:nth-child(9)  { animation-delay:0.45s; }
.usp-item:nth-child(10) { animation-delay:0.50s; }

/* Stagger for chips */
.amenity-chip:nth-child(1)  { animation-delay:0.02s; }
.amenity-chip:nth-child(2)  { animation-delay:0.04s; }
.amenity-chip:nth-child(3)  { animation-delay:0.06s; }
.amenity-chip:nth-child(4)  { animation-delay:0.08s; }
.amenity-chip:nth-child(5)  { animation-delay:0.10s; }
.amenity-chip:nth-child(6)  { animation-delay:0.12s; }
.amenity-chip:nth-child(7)  { animation-delay:0.14s; }
.amenity-chip:nth-child(8)  { animation-delay:0.16s; }
.amenity-chip:nth-child(9)  { animation-delay:0.18s; }
.amenity-chip:nth-child(10) { animation-delay:0.20s; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(196,160,100,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(196,160,100,0.4); }
</style>
""", unsafe_allow_html=True)

# ── API Key ────────────────────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🔑 GEMINI_API_KEY not found in secrets.")
    st.stop()

# ── Prompts ────────────────────────────────────────────────────────────────────
base_prompt = """You are an expert real estate copywriter specializing in premium residential properties. You have been provided a brochure PDF for a premium residential project.

Your task is to extract powerful, factual Unique Selling Propositions (USPs) that will compel high-net-worth buyers to take action.

EXTRACTION SCOPE — cover ALL of these angles:
1. Thematic & Architectural Identity — design philosophy, style, signature elements
2. Clubhouse & Lifestyle Amenities — name every facility with its size/count if stated
3. Technology, Automation & Security — smart home, surveillance, access control
4. Landscape, Green & Open Spaces — area %, acres, named gardens, water bodies
5. Location & Connectivity — distances to landmarks, roads, transit hubs (use exact km/min)
6. Developer, Architect & Consultant Pedigree — ONLY if a proper name is explicitly written
7. Awards, Certifications & Approvals — include certifying body and year if available
8. Unit & Project Specifications — total units, density, floors, BHK range, super area

QUALITY RULES:
• FACTUAL PRECISION: Every USP must reflect data explicitly present in the brochure.
• PROPER NOUNS ONLY: Include names ONLY if explicitly printed.
• NO NOISE: Ignore boilerplate marketing copy, generic slogans, disclaimers.
• NO HEADERS: Output is a flat list.
• RANKING: Most unique and buyer-influential USP first.

CHARACTER LIMIT — STRICTLY ENFORCED:
• Each USP text must be ≤ 75 characters (including spaces and punctuation).

CATEGORIZATION — assign exactly ONE:
AMENITIES, LOCATION_AND_CONNECTIVITY, CONSTRUCTION_AND_DESIGN,
TECHNOLOGY_AND_AUTOMATION, OFFERS, CERTIFICATES_AND_APPROVALS,
AWARDS_AND_ACCOLADES, MASTER_PLAN

OUTPUT FORMAT — one line per USP, nothing else:
[CATEGORY] | [SUB_CATEGORY] | [USP text ≤ 75 characters]

Begin extraction now. Output ONLY the formatted USP lines."""

old_usps_prompt = """
Additionally, I'm providing you with previously identified USPs for this property.

OLD USPs:
{old_usps}

Merge insights from both: remove duplicates, keep the most compelling points. Apply the same formatting, character limit, and quality rules to all final USPs.
"""

PREDEFINED_AMENITIES = [
    "Private Gardens/Balconies","Swimming Pool","Internal Street Lights","Gated Community",
    "Anti-termite Treatment","Earthquake Resistant","Paved Compound","Permeable Pavement",
    "Vastu Compliant","Wheelchair Accessible","Grade A Building","Feng Shui","Society Office",
    "Heli-Pad","Solar Lighting","Well-Maintained Internal Roads","Energy Efficient Lightining",
    "Community Hall","Solar Panel","Temple","School","Pet Park","Solar Water Heating",
    "Co-Working Spaces","Library","Carrom","Thermal Insulation","Creche/Day Care",
    "Outdoor Event Spaces","Air Hockey","Football Ground","Table Tennis","Volley Ball Court",
    "Pool Table","Chess","Dart Board","Billiards","Foosball","Cricket Pitch","Bowling Alley",
    "Lawn Tennis Court","Basketball Court","Rock Climbing Wall","Badminton Court",
    "Beach Volley Ball Court","Spa","Jacuzzi","Acupressure Park","Skating Rink","Squash Court",
    "Massage Room","Yoga/Meditation Area","Sauna","Futsal","Reflexology Park","Aerobics Centre",
    "Video Gaming Room","Ayurvedic Centre","Doctor on Call","Steam Room","Flower Garden",
    "Terrace Garden","Medical Centre","Gymnasium","Open Space","Landscape Garden","Fountain",
    "Clinic","Pilates Studios","Natural Pond","Pedestrian-Friendly Zones","Manicured Garden",
    "Senior Citizen Sitout","Archery Range","Water Park/Slides","Sit Out Area",
    "Community Garden/Urban Farming","Green Wall (Vertical Gardens)","Forest Trail",
    "Cabana Sitting","Park","Car Parking","Art and Craft Studio","EV Charging Stations",
    "Music Room","Dance Studio","Barbecue","Banquet Hall","Sun Deck","Party Lawn","Sand Pit",
    "Mini Theatre","Club House","Children's Play Area","Multipurpose Hall","Gazebo",
    "Amphitheatre","Card Room","Jogging Track","Multipurpose court","Theatre","Golf Course",
    "Tot Lot","Nature Trail","Theater Home","Cycling Track","Art Gallery","Fire Alarm",
    "Gaming Zones","Boom Barrier","Wine Cellar","Emergency Exits","Golf Simulator",
    "CCTV Camera Security","Golf Putty","Fire Fighting Systems","Security Cabin","Indoor Games",
    "Gas Leak Detectors","Biometric/Smart Card Access","Fire NOC","Video Door Security",
    "Theme Park","Smoke Detectors","24x7 Security","Panic Buttons in Apartments",
    "Rooftop Lounge","Car-Free Zones","Ambulance Service","Cigar Lounge","Intercom Facilities",
    "Emergency Evacuation Chairs","Signage and Road Markings","Lounge","Bar/Chill-Out Lounge",
    "Fall Detection Systems in Bathrooms","Defibrillators in Common Areas","Piped Gas",
    "Business Lounge","Restaurant","Waiting Lounge","Reading Lounge","Wi-Fi Connectivity",
    "Pergola","Smart Home Automation","DTH Television","Laundry","Conference Room",
    "Wi-Fi Zones in Common Areas","Cafeteria","RO System","Food Court","Laundromat",
    "Shopping Centre","Property Staff","Changing Area","Lifts","Name Plates",
    "Automated Car Wash","Concierge Service","Toilet for Drivers","Car Wash Area","Salon",
    "Grocery Shop","Bus Shelter","Milk Booth","Letter Box","Petrol Pump","Entrance Lobby",
    "24/7 Power Backup","Maintenance Staff","Intercom","ATM","DG Availability",
    "Power Back up Lift","Escalators","Noise Insulation in Apartments",
    "Centralized Air Conditioning","Plumber/Electrician on Call","Secretarial Services",
    "Underground Electric Cabling","Power Substation","Braille Signage","Air Purification Systems",
    "Composting Facilities","Recycling Facilities","Garbage Chute","Garbage Disposal",
    "Organic Waste Converter","Waste Segregation and Disposal","Waste Management",
    "Sewage Treatment Plant","Water Treatment Plant","Water Softener Plant","Smart Water Meters",
    "Rain Water Harvesting","Bioswales","Ground Water Recharging Systems","24/7 Water Supply",
    "Municipal Water Supply","Low Flow Fixtures","Greywater Recycling","Borewell Water Supply",
]

amenities_extraction_prompt = """You are an information extraction assistant.

Your task is to extract amenities mentioned in the provided real estate brochure PDF.

IMPORTANT RULES:
1. Only extract amenities that EXACTLY match items from the predefined amenities list below.
2. Do NOT add new amenities. Do NOT infer or assume amenities.
3. If something is similar but not an exact match, ignore it.
4. Return ONLY a JSON array — no preamble, no explanation, no markdown fences.
5. If no amenities from the list are found, return [].
6. Each amenity should appear only once. Preserve exact spelling from the list.

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

Output ONLY a valid JSON array. Example: ["Swimming Pool", "Gymnasium", "Club House"]"""

# ── Helpers ────────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "AMENITIES": "#c4a064",
    "LOCATION_AND_CONNECTIVITY": "#7ec4a0",
    "CONSTRUCTION_AND_DESIGN": "#a07ec4",
    "TECHNOLOGY_AND_AUTOMATION": "#7eaec4",
    "OFFERS": "#c47e9b",
    "CERTIFICATES_AND_APPROVALS": "#a4c47e",
    "AWARDS_AND_ACCOLADES": "#c4b47e",
    "MASTER_PLAN": "#7ec4c4",
}

def setup_gemini_client():
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return None

def download_pdf_from_url(url):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        if "application/pdf" not in ct and not url.lower().endswith(".pdf"):
            st.error(f"URL does not point to a valid PDF. Content-Type: {ct}")
            return None
        return r.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF: {e}")
        return None

def analyze_pdf_via_files_api(pdf_bytes, prompt, model_name, client):
    uploaded_gemini_file = None
    tmp_path = None
    try:
        with st.spinner("Uploading PDF to Gemini…"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            uploaded_gemini_file = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(mime_type="application/pdf"),
            )
            max_wait, waited = 60, 0
            while uploaded_gemini_file.state.name != "ACTIVE":
                if waited >= max_wait:
                    st.error("Gemini file processing timed out.")
                    return None
                time.sleep(3); waited += 3
                uploaded_gemini_file = client.files.get(name=uploaded_gemini_file.name)
        with st.spinner(f"Analyzing with {model_name}…"):
            result = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(file_uri=uploaded_gemini_file.uri, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt),
                ],
            )
        return result.text
    except Exception as e:
        st.error(f"Error during Gemini analysis: {e}")
        import traceback; st.error(traceback.format_exc())
        return None
    finally:
        if uploaded_gemini_file:
            try: client.files.delete(name=uploaded_gemini_file.name)
            except: pass
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-line"></div>
  <div class="hero-eyebrow">▸ AI-Powered Real Estate Intelligence</div>
  <div class="hero-title">Property Intel <span>Analyzer</span></div>
  <div class="hero-sub">Extract USPs &amp; Amenities from any brochure — powered by Google Gemini</div>
</div>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚙ Configuration</div>', unsafe_allow_html=True)
model_options = {
    "Gemini 3.1 Pro Preview": "gemini-3.1-pro-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro-preview-06-05",
}
selected_model_name = st.selectbox(
    "AI Model — switch if you encounter errors",
    options=list(model_options.keys()),
    index=0,
)
selected_model = model_options[selected_model_name]
st.markdown('</div>', unsafe_allow_html=True)

# ── Input row ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card"><div class="card-title">📄 Brochure Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)
    pdf_url = st.text_input("PDF URL", placeholder="https://example.com/brochure.pdf", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card"><div class="card-title">📝 Previous USPs (Optional)</div>', unsafe_allow_html=True)
    old_usps = st.text_area(
        "Old USPs",
        height=120,
        placeholder="AMENITIES | CLUB_HOUSE | ...\nLOCATION_AND_CONNECTIVITY | METRO | ...",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Resolve PDF ────────────────────────────────────────────────────────────────
pdf_bytes = None
input_source = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    input_source = uploaded_file.name
elif pdf_url and pdf_url.strip():
    pdf_bytes = download_pdf_from_url(pdf_url.strip())
    if pdf_bytes:
        input_source = pdf_url.strip()

# ── Results ────────────────────────────────────────────────────────────────────
if pdf_bytes:
    size_kb = len(pdf_bytes) / 1024
    st.markdown(
        f'<div style="margin-bottom:1.5rem">'
        f'<span class="pdf-pill">📎 {input_source} &nbsp;·&nbsp; {size_kb:.1f} KB</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_usp, tab_amenities = st.tabs(["✦  Extract USPs", "◈  Extract Amenities"])

    # ── USP Tab ────────────────────────────────────────────────────────────────
    with tab_usp:
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            run_usp = st.button("▸  Analyze Brochure", key="btn_usp", use_container_width=True)

        if run_usp:
            client = setup_gemini_client()
            if not client:
                st.stop()

            t0 = time.time()
            full_prompt = base_prompt
            if old_usps.strip():
                full_prompt += old_usps_prompt.format(old_usps=old_usps)

            analysis = analyze_pdf_via_files_api(pdf_bytes, full_prompt, selected_model, client)
            elapsed = time.time() - t0

            if analysis:
                usps_raw = [u.strip() for u in analysis.strip().split("\n") if u.strip()]
                parsed, cat_counts = [], {}
                for usp in usps_raw:
                    parts = [p.strip() for p in usp.split("|")]
                    if len(parts) == 3:
                        cat, sub, text = parts
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        parsed.append((cat, sub, text))
                    else:
                        parsed.append((None, None, usp))

                total = len(parsed)
                over_limit = sum(1 for _, _, t in parsed if t and len(t) > 75)
                unique_cats = len(cat_counts)

                st.markdown(f"""
                <div class="stats-bar">
                  <div class="stat-item"><div class="stat-num">{total}</div><div class="stat-label">Total USPs</div></div>
                  <div class="stat-item"><div class="stat-num">{unique_cats}</div><div class="stat-label">Categories</div></div>
                  <div class="stat-item"><div class="stat-num">{over_limit}</div><div class="stat-label">Over Limit</div></div>
                  <div class="stat-item"><div class="stat-num">{elapsed:.0f}s</div><div class="stat-label">Time Taken</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-heading">Extracted USPs</div>', unsafe_allow_html=True)

                for cat, sub, text in parsed:
                    if cat:
                        color = CATEGORY_COLORS.get(cat, "#c4a064")
                        char_count = len(text)
                        over_warn = f'<div class="usp-warn">⚠ {char_count} chars — exceeds 75 limit</div>' if char_count > 75 else ""
                        st.markdown(f"""
                        <div class="usp-item">
                          <div><span class="usp-badge" style="color:{color};background:{color}18;border:1px solid {color}35;">{cat}</span></div>
                          <div style="flex:1">
                            <div class="usp-sub">{sub}</div>
                            <div class="usp-text">{text}</div>
                            {over_warn}
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="usp-item"><div class="usp-text">{text}</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="timing-note">✓ Completed in {elapsed:.1f}s · {selected_model_name} · PDF deleted from Gemini after processing</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button("⬇  Download USPs (.txt)", data=analysis, file_name="property_usps.txt", mime="text/plain")
            else:
                st.error("Analysis failed. Please try again or switch models.")

    # ── Amenities Tab ──────────────────────────────────────────────────────────
    with tab_amenities:
        st.markdown(
            f'<p style="font-size:0.82rem;color:#6b6456;margin-bottom:1.2rem;">'
            f'Matches against {len(PREDEFINED_AMENITIES)} predefined amenities exactly.</p>',
            unsafe_allow_html=True,
        )
        btn_col2, _ = st.columns([1, 3])
        with btn_col2:
            run_am = st.button("◈  Extract Amenities", key="btn_amenities", use_container_width=True)

        if run_am:
            client = setup_gemini_client()
            if not client:
                st.stop()

            t0 = time.time()
            raw = analyze_pdf_via_files_api(pdf_bytes, amenities_extraction_prompt, selected_model, client)
            elapsed = time.time() - t0

            if raw:
                try:
                    clean = raw.strip().strip("```json").strip("```").strip()
                    extracted = json.loads(clean)
                except json.JSONDecodeError:
                    st.error("Unexpected format from model. Raw response:")
                    st.code(raw)
                    extracted = []

                valid = [a for a in extracted if a in PREDEFINED_AMENITIES]
                invalid = [a for a in extracted if a not in PREDEFINED_AMENITIES]

                if valid:
                    pct = round(len(valid) / len(PREDEFINED_AMENITIES) * 100)
                    st.markdown(f"""
                    <div class="stats-bar">
                      <div class="stat-item"><div class="stat-num">{len(valid)}</div><div class="stat-label">Found</div></div>
                      <div class="stat-item"><div class="stat-num">{len(PREDEFINED_AMENITIES)}</div><div class="stat-label">Total List</div></div>
                      <div class="stat-item"><div class="stat-num">{pct}%</div><div class="stat-label">Coverage</div></div>
                      <div class="stat-item"><div class="stat-num">{elapsed:.0f}s</div><div class="stat-label">Time Taken</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-heading">Amenities Found</div>', unsafe_allow_html=True)

                    chips = "".join(
                        f'<span class="amenity-chip"><span class="dot"></span>{a}</span>'
                        for a in valid
                    )
                    st.markdown(f'<div style="line-height:2.6">{chips}</div>', unsafe_allow_html=True)

                    if invalid:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.warning(f"⚠ {len(invalid)} item(s) did not match the predefined list and were excluded: {', '.join(invalid)}")

                    st.markdown(f'<div class="timing-note">✓ Completed in {elapsed:.1f}s · {selected_model_name} · PDF deleted from Gemini after processing</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    dl1, dl2, _ = st.columns([1, 1, 2])
                    with dl1:
                        st.download_button("⬇  JSON", data=json.dumps(valid, indent=2), file_name="property_amenities.json", mime="application/json")
                    with dl2:
                        st.download_button("⬇  TXT", data="\n".join(valid), file_name="property_amenities.txt", mime="text/plain")
                else:
                    st.info("No matching amenities were found in this brochure.")
                    st.markdown(f'<div class="timing-note">✓ Completed in {elapsed:.1f}s · {selected_model_name}</div>', unsafe_allow_html=True)
            else:
                st.error("Extraction failed. Please try again or switch models.")

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#3a3530;">
      <div style="font-size:3rem;margin-bottom:1rem;opacity:0.35">🏛</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.75rem;letter-spacing:0.2em;text-transform:uppercase;">
        Upload a brochure PDF or paste a URL above to begin
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-caption">
  PROPERTY INTEL ANALYZER &nbsp;·&nbsp; POWERED BY GOOGLE GEMINI &nbsp;·&nbsp; BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)
