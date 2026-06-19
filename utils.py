"""
utils.py  —  Shared helpers for Indian House Price Prediction App
"""

import numpy as np
import pandas as pd

# ── Indian city localities ────────────────────────────────────────────────────
CITIES = [
    "Mumbai - Bandra",
    "Mumbai - Andheri",
    "Mumbai - Thane",
    "Delhi - South Delhi",
    "Delhi - Dwarka",
    "Noida - Sector 62",
    "Gurgaon - DLF Phase",
    "Bengaluru - Koramangala",
    "Bengaluru - Whitefield",
    "Bengaluru - Sarjapur",
    "Hyderabad - Gachibowli",
    "Hyderabad - Banjara Hills",
    "Pune - Koregaon Park",
    "Pune - Hinjewadi",
    "Pune - Viman Nagar",
    "Chennai - Adyar",
    "Chennai - OMR",
    "Kolkata - Salt Lake",
    "Kolkata - New Town",
    "Ahmedabad - SG Highway",
    "Jaipur - Malviya Nagar",
    "Kochi - Marine Drive",
]

CITY_EMOJIS = {
    "Mumbai - Bandra":            "🌊",
    "Mumbai - Andheri":           "🏙️",
    "Mumbai - Thane":             "🏘️",
    "Delhi - South Delhi":        "🏛️",
    "Delhi - Dwarka":             "🏗️",
    "Noida - Sector 62":          "💻",
    "Gurgaon - DLF Phase":        "🏢",
    "Bengaluru - Koramangala":    "☕",
    "Bengaluru - Whitefield":     "🖥️",
    "Bengaluru - Sarjapur":       "🌿",
    "Hyderabad - Gachibowli":     "💡",
    "Hyderabad - Banjara Hills":  "👑",
    "Pune - Koregaon Park":       "🌳",
    "Pune - Hinjewadi":           "⚙️",
    "Pune - Viman Nagar":         "✈️",
    "Chennai - Adyar":            "🌴",
    "Chennai - OMR":              "🔬",
    "Kolkata - Salt Lake":        "🏫",
    "Kolkata - New Town":         "🆕",
    "Ahmedabad - SG Highway":     "🛣️",
    "Jaipur - Malviya Nagar":     "🏰",
    "Kochi - Marine Drive":       "⛵",
}

CITY_INFO = {
    "Mumbai - Bandra":            "Premium sea-facing locality; Bollywood hub.",
    "Mumbai - Andheri":           "Well-connected suburb; commercial & residential mix.",
    "Mumbai - Thane":             "Affordable Mumbai alternative; rapid development.",
    "Delhi - South Delhi":        "Most premium Delhi locality; diplomatic enclave zone.",
    "Delhi - Dwarka":             "Planned sub-city; great metro connectivity.",
    "Noida - Sector 62":          "IT & corporate park zone; modern infrastructure.",
    "Gurgaon - DLF Phase":        "Millennium City; MNC offices & luxury apartments.",
    "Bengaluru - Koramangala":    "Startup capital's hotspot; vibrant cafe culture.",
    "Bengaluru - Whitefield":     "Major IT corridor; expat-friendly locality.",
    "Bengaluru - Sarjapur":       "Emerging tech suburb; good appreciation potential.",
    "Hyderabad - Gachibowli":     "HITEC City neighbor; top IT companies.",
    "Hyderabad - Banjara Hills":  "Upscale Hyderabad; luxury residences.",
    "Pune - Koregaon Park":       "Upmarket Pune; expat & business community.",
    "Pune - Hinjewadi":           "IT park hub; Infosys, Wipro offices nearby.",
    "Pune - Viman Nagar":         "Near airport; cosmopolitan neighborhood.",
    "Chennai - Adyar":            "Old-money neighborhood; calm & green.",
    "Chennai - OMR":              "Old Mahabalipuram Road; Chennai's IT spine.",
    "Kolkata - Salt Lake":        "Planned township; IT sector & government offices.",
    "Kolkata - New Town":         "Smartcity project; modern infrastructure.",
    "Ahmedabad - SG Highway":     "Sarkhej-Gandhinagar corridor; rapid growth.",
    "Jaipur - Malviya Nagar":     "Pink City's upscale zone; wide roads.",
    "Kochi - Marine Drive":       "Kerala's waterfront gem; premium pricing.",
}

CITY_PRICE_MULT = {
    "Mumbai - Bandra":            2.80,
    "Mumbai - Andheri":           2.20,
    "Mumbai - Thane":             1.60,
    "Delhi - South Delhi":        2.50,
    "Delhi - Dwarka":             1.80,
    "Noida - Sector 62":          1.40,
    "Gurgaon - DLF Phase":        2.00,
    "Bengaluru - Koramangala":    2.10,
    "Bengaluru - Whitefield":     1.75,
    "Bengaluru - Sarjapur":       1.55,
    "Hyderabad - Gachibowli":     1.65,
    "Hyderabad - Banjara Hills":  1.90,
    "Pune - Koregaon Park":       1.70,
    "Pune - Hinjewadi":           1.35,
    "Pune - Viman Nagar":         1.50,
    "Chennai - Adyar":            1.80,
    "Chennai - OMR":              1.40,
    "Kolkata - Salt Lake":        1.45,
    "Kolkata - New Town":         1.30,
    "Ahmedabad - SG Highway":     1.25,
    "Jaipur - Malviya Nagar":     1.20,
    "Kochi - Marine Drive":       1.55,
}

FURNISHING_OPTIONS = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FLOOR_OPTIONS      = ["Ground", "Low (1-4)", "Mid (5-10)", "High (11+)"]


def format_price(price_lakh: float) -> str:
    """Format price in Indian currency notation."""
    if price_lakh >= 100:
        cr = price_lakh / 100
        return f"Rs. {cr:.2f} Cr"
    return f"Rs. {price_lakh:.2f} L"


def price_range(price_lakh: float, pct: float = 0.08):
    lo = price_lakh * (1 - pct)
    hi = price_lakh * (1 + pct)
    return format_price(lo), format_price(hi)


def age_label(age: int) -> str:
    if age == 0:   return "Brand New / Under Construction"
    if age <= 3:   return "Nearly New (< 3 yrs)"
    if age <= 8:   return "Modern (< 8 yrs)"
    if age <= 15:  return "Established (< 15 yrs)"
    return "Old Property (15+ yrs)"


def build_input_df(area, bedrooms, bathrooms, parking, age,
                   furnishing, floor, city) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame(
        [[area, bedrooms, bathrooms, parking, age, furnishing, floor, city]],
        columns=["Area", "Bedrooms", "Bathrooms", "Parking", "Age",
                 "Furnishing", "Floor", "City"],
    )
