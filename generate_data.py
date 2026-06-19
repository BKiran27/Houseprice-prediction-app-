"""
generate_data.py
----------------
Generates a realistic synthetic Indian house price dataset.
Cities: Mumbai, Delhi, Bengaluru, Hyderabad, Pune, Chennai,
        Kolkata, Noida, Gurgaon, Ahmedabad, Jaipur, Kochi
Run once before train.py.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1500

# Indian city tiers with realistic price multipliers (base = Pune suburbs)
CITY_DATA = {
    "Mumbai - Bandra":       2.80,
    "Mumbai - Andheri":      2.20,
    "Mumbai - Thane":        1.60,
    "Delhi - South Delhi":   2.50,
    "Delhi - Dwarka":        1.80,
    "Noida - Sector 62":     1.40,
    "Gurgaon - DLF Phase":   2.00,
    "Bengaluru - Koramangala":2.10,
    "Bengaluru - Whitefield": 1.75,
    "Bengaluru - Sarjapur":  1.55,
    "Hyderabad - Gachibowli":1.65,
    "Hyderabad - Banjara Hills":1.90,
    "Pune - Koregaon Park":  1.70,
    "Pune - Hinjewadi":      1.35,
    "Pune - Viman Nagar":    1.50,
    "Chennai - Adyar":       1.80,
    "Chennai - OMR":         1.40,
    "Kolkata - Salt Lake":   1.45,
    "Kolkata - New Town":    1.30,
    "Ahmedabad - SG Highway":1.25,
    "Jaipur - Malviya Nagar":1.20,
    "Kochi - Marine Drive":  1.55,
}

city_names = list(CITY_DATA.keys())
city_mults = list(CITY_DATA.values())

# Probability weights (weighted toward major metros)
city_probs = [
    0.07, 0.06, 0.05,   # Mumbai
    0.06, 0.05, 0.05,   # Delhi/Noida
    0.05,               # Gurgaon
    0.07, 0.06, 0.05,   # Bengaluru
    0.05, 0.04,         # Hyderabad
    0.05, 0.04, 0.04,   # Pune
    0.04, 0.04,         # Chennai
    0.04, 0.03,         # Kolkata
    0.04,               # Ahmedabad
    0.04,               # Jaipur
    0.03,               # Kochi
]
# Normalize probabilities
city_probs = [p / sum(city_probs) for p in city_probs]

locations  = np.random.choice(city_names, size=N, p=city_probs)
bedrooms   = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.10, 0.25, 0.38, 0.20, 0.07])
bathrooms  = np.clip(bedrooms - np.random.choice([0, 1], size=N, p=[0.65, 0.35]), 1, 5)
parking    = np.random.choice([0, 1, 2, 3], size=N, p=[0.12, 0.48, 0.30, 0.10])
age        = np.random.randint(0, 35, size=N)

# Furnishing status (affects price)
furnishing = np.random.choice(["Unfurnished", "Semi-Furnished", "Fully Furnished"],
                               size=N, p=[0.30, 0.45, 0.25])
furnish_mult = {"Unfurnished": 0.90, "Semi-Furnished": 1.00, "Fully Furnished": 1.12}

# Floor type
floor_type = np.random.choice(["Ground", "Low (1-4)", "Mid (5-10)", "High (11+)"],
                               size=N, p=[0.15, 0.35, 0.30, 0.20])
floor_mult = {"Ground": 0.95, "Low (1-4)": 1.00, "Mid (5-10)": 1.04, "High (11+)": 1.08}

# Area correlated with bedrooms (sq ft — standard Indian sizing)
base_area  = bedrooms * 400
area       = (base_area + np.random.normal(0, 180, size=N)).clip(350, 5500).astype(int)

# Base price in Lakhs — realistic Indian market
BASE = 20.0
price = (
    BASE
    + area        * 0.030
    + bedrooms    * 4.0
    + bathrooms   * 3.0
    + parking     * 2.5
    - age         * 0.40
    + np.random.normal(0, 5, size=N)
) * np.array([CITY_DATA[l] for l in locations]) \
  * np.array([furnish_mult[f] for f in furnishing]) \
  * np.array([floor_mult[f] for f in floor_type])

price = price.clip(10, 800).round(2)   # 10 Lakh – 8 Cr

df = pd.DataFrame({
    "Area":       area,
    "Bedrooms":   bedrooms,
    "Bathrooms":  bathrooms.astype(int),
    "Parking":    parking,
    "Age":        age,
    "Furnishing": furnishing,
    "Floor":      floor_type,
    "City":       locations,
    "Price":      price,   # in Lakhs
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/house_data.csv", index=False)
print(f"[OK] Dataset saved -> data/house_data.csv  ({len(df)} rows)")
print(df.describe(include="all").round(2).to_string())
