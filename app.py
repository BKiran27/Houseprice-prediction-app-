"""
app.py  —  Ghar Bazaar: Indian House Price Prediction App
Tabs: Predict | Data Explorer | Model Performance | Compare Properties
Compatible with Streamlit 1.58+ | Auto-bootstraps model on first run
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    CITIES, CITY_EMOJIS, CITY_INFO, CITY_PRICE_MULT,
    FURNISHING_OPTIONS, FLOOR_OPTIONS,
    format_price, price_range, age_label, build_input_df,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ghar Bazaar | Indian House Price AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0533 45%, #0d1b2a 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(255,153,0,0.18) 0%, rgba(19,136,8,0.12) 50%, rgba(6,6,180,0.18) 100%);
    border: 1px solid rgba(255,153,0,0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff9900, #ffffff, #138808);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero .tagline { color: #94a3b8; font-size: 1rem; margin: 0; }
.hero .flag { font-size: 1.5rem; margin-bottom: 0.4rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(255,153,0,0.15); }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,153,0,0.12), rgba(19,136,8,0.10));
    border: 1px solid rgba(255,153,0,0.25);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; font-weight: 700; letter-spacing: .08em;
                text-transform: uppercase; color: #94a3b8; margin-bottom: .25rem; }
.metric-value { font-family: 'Outfit', sans-serif; font-size: 1.55rem;
                font-weight: 700; color: #f1f5f9; }
.metric-sub   { font-size: 0.76rem; color: #64748b; margin-top: .15rem; }

/* Predict result box */
.predict-box {
    background: linear-gradient(135deg, #c05800 0%, #1a6b00 50%, #000087 100%);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(255,153,0,0.35);
    animation: saffron-glow 3s ease-in-out infinite;
}
@keyframes saffron-glow {
    0%,100% { box-shadow: 0 20px 60px rgba(255,153,0,0.35); }
    50%      { box-shadow: 0 20px 80px rgba(255,153,0,0.60); }
}
.predict-price { font-family: 'Outfit',sans-serif; font-size:3rem; font-weight:800; color:#fff; line-height:1.1; }
.predict-label { font-size:.82rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:rgba(255,255,255,.6); margin-bottom:.6rem; }
.predict-range { font-size:.95rem; color:rgba(255,255,255,.7); margin-top:.5rem; }

/* Section headers */
.section-header { font-family:'Outfit',sans-serif; font-size:1.35rem; font-weight:700;
                  color:#f1f5f9; margin:1.4rem 0 .7rem; }

/* Tags */
.tag { display:inline-block; background:rgba(255,153,0,.15); border:1px solid rgba(255,153,0,.3);
       border-radius:99px; padding:.18rem .7rem; font-size:.78rem; font-weight:500;
       color:#ffb347; margin:.12rem; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] { font-weight:600; color:#94a3b8 !important;
                                      border-radius:10px 10px 0 0; padding:.55rem 1.1rem; }
[data-testid="stTabs"] [aria-selected="true"] { color:#ff9900 !important;
    background:rgba(255,153,0,.10) !important; border-bottom:2px solid #ff9900 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff9900, #138808);
    color: white; font-weight: 700; font-size: 1rem;
    border: none; border-radius: 12px; padding: .7rem 2rem;
    width: 100%; transition: all .2s;
    box-shadow: 0 4px 15px rgba(255,153,0,.35);
}
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(255,153,0,.5); }

hr { border-color:rgba(255,255,255,.07) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,153,0,.4); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme (NO margin here — override per chart) ────────────────────────
PLOT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#94a3b8"),
    title_font=dict(family="Outfit,sans-serif", color="#f1f5f9", size=15),
    colorway=["#ff9900","#138808","#0606b4","#e94560","#00b4d8","#9b5de5","#f15bb5"],
)
PLOT_MARGIN = dict(l=20, r=20, t=45, b=20)


def apply_layout(fig, **extra):
    fig.update_layout(**PLOT_BASE, margin=PLOT_MARGIN, **extra)


# ═══════════════════════════════════════════════════════════════════════════════
# Load model & data
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Bootstrap: generate data + train model if not present (Streamlit Cloud)
# ═══════════════════════════════════════════════════════════════════════════════
def auto_bootstrap():
    """Run data generation + training inline if artifacts are missing."""
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # ── Generate data ──────────────────────────────────────────────────────────
    if not os.path.exists("data/house_data.csv"):
        np.random.seed(42)
        N = 1500
        CITY_DATA = {
            "Mumbai - Bandra":2.80,"Mumbai - Andheri":2.20,"Mumbai - Thane":1.60,
            "Delhi - South Delhi":2.50,"Delhi - Dwarka":1.80,
            "Noida - Sector 62":1.40,"Gurgaon - DLF Phase":2.00,
            "Bengaluru - Koramangala":2.10,"Bengaluru - Whitefield":1.75,"Bengaluru - Sarjapur":1.55,
            "Hyderabad - Gachibowli":1.65,"Hyderabad - Banjara Hills":1.90,
            "Pune - Koregaon Park":1.70,"Pune - Hinjewadi":1.35,"Pune - Viman Nagar":1.50,
            "Chennai - Adyar":1.80,"Chennai - OMR":1.40,
            "Kolkata - Salt Lake":1.45,"Kolkata - New Town":1.30,
            "Ahmedabad - SG Highway":1.25,"Jaipur - Malviya Nagar":1.20,"Kochi - Marine Drive":1.55,
        }
        city_names = list(CITY_DATA.keys())
        city_probs = [0.07,0.06,0.05,0.06,0.05,0.05,0.05,0.07,0.06,0.05,
                      0.05,0.04,0.05,0.04,0.04,0.04,0.04,0.04,0.03,0.04,0.04,0.03]
        city_probs = [p/sum(city_probs) for p in city_probs]
        locations  = np.random.choice(city_names, size=N, p=city_probs)
        bedrooms   = np.random.choice([1,2,3,4,5], size=N, p=[0.10,0.25,0.38,0.20,0.07])
        bathrooms  = np.clip(bedrooms - np.random.choice([0,1], size=N, p=[0.65,0.35]),1,5)
        parking    = np.random.choice([0,1,2,3], size=N, p=[0.12,0.48,0.30,0.10])
        age        = np.random.randint(0, 35, size=N)
        furnishing = np.random.choice(["Unfurnished","Semi-Furnished","Fully Furnished"],
                                      size=N, p=[0.30,0.45,0.25])
        floor_type = np.random.choice(["Ground","Low (1-4)","Mid (5-10)","High (11+)"],
                                      size=N, p=[0.15,0.35,0.30,0.20])
        base_area  = bedrooms * 400
        area       = (base_area + np.random.normal(0,180,size=N)).clip(350,5500).astype(int)
        furn_mult  = {"Unfurnished":0.90,"Semi-Furnished":1.00,"Fully Furnished":1.12}
        flr_mult   = {"Ground":0.95,"Low (1-4)":1.00,"Mid (5-10)":1.04,"High (11+)":1.08}
        price = (
            20.0 + area*0.030 + bedrooms*4.0 + bathrooms*3.0 + parking*2.5
            - age*0.40 + np.random.normal(0,5,size=N)
        ) * np.array([CITY_DATA[l] for l in locations]) \
          * np.array([furn_mult[f] for f in furnishing]) \
          * np.array([flr_mult[f]  for f in floor_type])
        price = price.clip(10, 800).round(2)
        df_gen = pd.DataFrame({
            "Area":area,"Bedrooms":bedrooms,"Bathrooms":bathrooms.astype(int),
            "Parking":parking,"Age":age,"Furnishing":furnishing,
            "Floor":floor_type,"City":locations,"Price":price,
        })
        os.makedirs("data", exist_ok=True)
        df_gen.to_csv("data/house_data.csv", index=False)

    # ── Train models ───────────────────────────────────────────────────────────
    df_t = pd.read_csv("data/house_data.csv")
    FEATURES = ["Area","Bedrooms","Bathrooms","Parking","Age","Furnishing","Floor","City"]
    X = df_t[FEATURES];  y = df_t["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_f = ["Area","Bedrooms","Bathrooms","Parking","Age"]
    cat_f = ["Furnishing","Floor","City"]
    pre   = ColumnTransformer([
        ("num", StandardScaler(),                       num_f),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_f),
    ])
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    results, pipelines = {}, {}
    for name, reg in models.items():
        pipe = Pipeline([("preprocessor", pre), ("regressor", reg)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        cv    = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
        results[name] = {
            "MAE":  mean_absolute_error(y_test, preds),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "R2":   r2_score(y_test, preds),
            "CV_R2": cv,
        }
        pipelines[name] = pipe

    best = max(results, key=lambda k: results[k]["CV_R2"])
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": pipelines[best], "model_name": best,
        "all_results": results, "features": FEATURES,
        "all_pipelines": pipelines,
    }, "models/model.pkl")


# ── Run bootstrap silently if model missing ────────────────────────────────────
if not os.path.exists("models/model.pkl"):
    with st.spinner("Setting up AI model for first time... Please wait ~30 seconds"):
        auto_bootstrap()
    st.cache_resource.clear()
    st.cache_data.clear()


@st.cache_resource(show_spinner="Loading AI model ...")
def load_model():
    path = "models/model.pkl"
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    path = "data/house_data.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


payload     = load_model()
df          = load_data()
model_ready = payload is not None
data_ready  = df is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.5rem;'>🏡</div>
        <div style='font-family:Outfit,sans-serif;font-size:1.35rem;font-weight:800;
                    background:linear-gradient(90deg,#ff9900,#ffffff,#138808);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Ghar Bazaar
        </div>
        <div style='font-size:.76rem;color:#64748b;margin-top:.2rem;'>
            Indian Real Estate AI
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if model_ready:
        mn = payload["model_name"]
        r  = payload["all_results"][mn]
        st.markdown("**Active Model**")
        st.success(mn)
        c1, c2 = st.columns(2)
        c1.metric("R2 Score", f"{r['R2']:.3f}")
        c2.metric("MAE", f"Rs.{r['MAE']:.1f}L")
        st.markdown("---")
    else:
        st.warning("Model not trained.\nRun:\n```\npython generate_data.py\npython train.py\n```")
        st.markdown("---")

    st.markdown("**Quick Guide**")
    st.markdown("""
    <div style='font-size:.84rem;color:#94a3b8;line-height:1.9;'>
    1. <b>Predict</b> — Enter property details<br>
    2. <b>Data Explorer</b> — Visualize dataset<br>
    3. <b>Model Performance</b> — Compare 3 models<br>
    4. <b>Compare</b> — Side-by-side analysis
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.73rem;color:#475569;text-align:center;'>
    Streamlit · Scikit-learn · Plotly<br>
    22 Indian City Localities · 1500 Records
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Hero
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="flag">🇮🇳</div>
    <h1>Ghar Bazaar — Indian House Price AI</h1>
    <p class="tagline">Predict property prices across 22 Indian city localities · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏡  Predict Price",
    "📊  Data Explorer",
    "📈  Model Performance",
    "🔍  Compare Properties",
])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PREDICT                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab1:
    if not model_ready:
        st.warning("Run `python generate_data.py` then `python train.py` to get started.")
        st.stop()

    st.markdown('<div class="section-header">🏡 Enter Property Details</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            area = st.slider("Area (sq ft)", 350, 5500, 1200, step=50,
                             help="Super built-up area in square feet")
        with r1c2:
            age = st.slider("Property Age (yrs)", 0, 35, 3,
                            help="0 = Under construction / brand new")

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            bedrooms  = st.selectbox("Bedrooms",  [1, 2, 3, 4, 5], index=2)
        with r2c2:
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
        with r2c3:
            parking   = st.selectbox("Parking",   [0, 1, 2, 3],    index=1)

        furnishing = st.selectbox("Furnishing",
                                  FURNISHING_OPTIONS,
                                  index=1,
                                  help="Affects ~10-12% price difference")

        floor = st.selectbox("Floor",
                             FLOOR_OPTIONS,
                             index=1,
                             help="Higher floors command a premium in most Indian cities")

        city = st.selectbox(
            "City / Locality",
            CITIES,
            format_func=lambda c: f"{CITY_EMOJIS[c]}  {c}",
        )
        st.markdown(
            f'<div style="font-size:.82rem;color:#64748b;margin-top:-.35rem;margin-bottom:.5rem;">'
            f'{CITY_INFO[city]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict House Price", use_container_width=True)  # button OK

    with col_right:
        # Property summary tags
        st.markdown('<div class="section-header">Property Summary</div>', unsafe_allow_html=True)
        tags_html = (
            f'<span class="tag">Area: {area:,} sqft</span>'
            f'<span class="tag">BHK: {bedrooms}</span>'
            f'<span class="tag">Bath: {bathrooms}</span>'
            f'<span class="tag">Park: {parking}</span>'
            f'<span class="tag">{age_label(age)}</span>'
            f'<span class="tag">{furnishing}</span>'
            f'<span class="tag">{floor} Floor</span>'
            f'<span class="tag">{CITY_EMOJIS[city]} {city.split(" - ")[0]}</span>'
        )
        st.markdown(f'<div class="card">{tags_html}</div>', unsafe_allow_html=True)

        if predict_btn or "last_pred" in st.session_state:
            if predict_btn:
                inp  = build_input_df(area, bedrooms, bathrooms, parking, age,
                                      furnishing, floor, city)
                pred = payload["model"].predict(inp)[0]
                st.session_state["last_pred"] = pred

            pred = st.session_state["last_pred"]
            lo, hi = price_range(pred)
            ppsf   = (pred * 1e5) / area

            st.markdown(f"""
            <div class="predict-box" style="margin-top:1.2rem;">
                <div class="predict-label">Estimated Market Value</div>
                <div class="predict-price">{format_price(pred)}</div>
                <div class="predict-range">Range: {lo} &mdash; {hi}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            for col, lbl, val in [
                (m1, "Price / Sq Ft", f"Rs.{ppsf:,.0f}"),
                (m2, "Model", payload["model_name"].split()[0]),
                (m3, "R2 Score", f"{payload['all_results'][payload['model_name']]['R2']:.3f}"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{val}</div>
                </div>""", unsafe_allow_html=True)

            # Feature radar
            st.markdown('<div class="section-header">Feature Influence</div>', unsafe_allow_html=True)
            mults = list(CITY_PRICE_MULT.values())
            city_mult = CITY_PRICE_MULT[city]
            norm_loc  = (city_mult - min(mults)) / (max(mults) - min(mults))
            furnish_score = {"Unfurnished": 0.3, "Semi-Furnished": 0.6, "Fully Furnished": 1.0}
            floor_score   = {"Ground": 0.4, "Low (1-4)": 0.6, "Mid (5-10)": 0.8, "High (11+)": 1.0}

            cats = ["Area", "BHK", "Bathrooms", "Age Factor", "Location", "Furnishing", "Floor"]
            vals = [
                min(area / 5500, 1.0),
                bedrooms / 5,
                bathrooms / 5,
                1 - age / 35,
                norm_loc,
                furnish_score[furnishing],
                floor_score[floor],
            ]

            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(255,153,0,0.20)",
                line=dict(color="#ff9900", width=2),
                marker=dict(size=6, color="#ffb347"),
            ))
            fig_radar.update_layout(
                **PLOT_BASE,
                margin=dict(l=35, r=35, t=18, b=18),
                polar=dict(
                    bgcolor="rgba(255,255,255,0.03)",
                    radialaxis=dict(visible=True, range=[0, 1.1],
                                   gridcolor="rgba(255,255,255,0.08)",
                                   tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                     tickfont=dict(size=10, color="#94a3b8")),
                ),
                showlegend=False,
                height=280,
            )
            st.plotly_chart(fig_radar, width="stretch")

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;margin-top:1.2rem;">
                <div style="font-size:3rem;margin-bottom:.7rem;">🔮</div>
                <div style="font-family:Outfit,sans-serif;font-size:1.15rem;font-weight:600;color:#f1f5f9;">
                    Your prediction will appear here
                </div>
                <div style="font-size:.87rem;color:#64748b;margin-top:.35rem;">
                    Fill in property details and click Predict
                </div>
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — DATA EXPLORER                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab2:
    if not data_ready:
        st.warning("Run `python generate_data.py` first.")
        st.stop()

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Records",  f"{len(df):,}",               "properties"),
        ("Avg Area",       f"{df.Area.mean():,.0f} sqft", ""),
        ("Avg Price",      format_price(df.Price.mean()), "market avg"),
        ("Max Price",      format_price(df.Price.max()),  "premium listing"),
        ("Cities",         "22",                          "Indian localities"),
    ]
    for col, (lbl, val, sub) in zip([k1, k2, k3, k4, k5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="font-size:1.2rem;">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Price histogram + median by city
    r1a, r1b = st.columns(2, gap="medium")

    with r1a:
        fig1 = px.histogram(df, x="Price", nbins=60,
                            title="Price Distribution (Lakhs)",
                            labels={"Price": "Price (Lakh)"},
                            color_discrete_sequence=["#ff9900"])
        fig1.update_traces(opacity=0.85)
        apply_layout(fig1)
        st.plotly_chart(fig1, width="stretch")

    with r1b:
        # Extract metro from city name for cleaner axis
        df["Metro"] = df["City"].apply(lambda x: x.split(" - ")[0])
        metro_med   = df.groupby("Metro")["Price"].median().reset_index().sort_values("Price")
        fig2 = px.bar(metro_med, x="Price", y="Metro", orientation="h",
                      title="Median Price by Metro City",
                      labels={"Price": "Median Price (Lakh)", "Metro": ""},
                      color="Price",
                      color_continuous_scale=["#138808", "#ff9900", "#e94560"])
        apply_layout(fig2, coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

    # Row 2: Area vs Price + BHK box
    r2a, r2b = st.columns(2, gap="medium")

    with r2a:
        fig3 = px.scatter(df, x="Area", y="Price", color="Metro",
                          title="Area vs Price by Metro City",
                          labels={"Area": "Area (sq ft)", "Price": "Price (Lakh)"},
                          opacity=0.6,
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        apply_layout(fig3)
        st.plotly_chart(fig3, width="stretch")

    with r2b:
        fig4 = px.box(df, x="Bedrooms", y="Price",
                      title="Price by BHK Configuration",
                      labels={"Bedrooms": "Bedrooms (BHK)", "Price": "Price (Lakh)"},
                      color="Bedrooms",
                      color_discrete_sequence=["#ff9900","#138808","#0606b4","#e94560","#9b5de5"])
        apply_layout(fig4, showlegend=False)
        st.plotly_chart(fig4, width="stretch")

    # Row 3: Furnishing + floor analysis
    r3a, r3b = st.columns(2, gap="medium")

    with r3a:
        furn_med = df.groupby("Furnishing")["Price"].median().reset_index()
        fig5 = px.bar(furn_med, x="Furnishing", y="Price",
                      title="Median Price by Furnishing Status",
                      labels={"Price": "Median Price (Lakh)"},
                      color="Furnishing",
                      color_discrete_sequence=["#64748b","#ff9900","#138808"])
        apply_layout(fig5, showlegend=False)
        st.plotly_chart(fig5, width="stretch")

    with r3b:
        floor_med = df.groupby("Floor")["Price"].median().reset_index()
        fig6 = px.bar(floor_med, x="Floor", y="Price",
                      title="Median Price by Floor Type",
                      labels={"Price": "Median Price (Lakh)"},
                      color="Price",
                      color_continuous_scale=["#138808","#ff9900","#e94560"])
        apply_layout(fig6, coloraxis_showscale=False)
        st.plotly_chart(fig6, width="stretch")

    # Row 4: Correlation + Age vs Price
    r4a, r4b = st.columns(2, gap="medium")

    with r4a:
        num_cols = ["Area", "Bedrooms", "Bathrooms", "Parking", "Age", "Price"]
        corr = df[num_cols].corr()
        fig7 = px.imshow(corr, title="Feature Correlation Matrix",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, text_auto=".2f")
        apply_layout(fig7)
        st.plotly_chart(fig7, width="stretch")

    with r4b:
        z = np.polyfit(df["Age"], df["Price"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["Age"].min(), df["Age"].max(), 100)
        fig8 = px.scatter(df, x="Age", y="Price", color="Metro",
                          title="Property Age vs Price",
                          labels={"Age": "Age (Years)", "Price": "Price (Lakh)"},
                          opacity=0.5,
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        fig8.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines",
                                  line=dict(color="#ff9900", width=2.5, dash="dash"),
                                  name="Trend"))
        apply_layout(fig8)
        st.plotly_chart(fig8, width="stretch")

    # Top 10 expensive localities
    st.markdown('<div class="section-header">Most Expensive Localities</div>', unsafe_allow_html=True)
    city_med = df.groupby("City")["Price"].median().reset_index().sort_values("Price", ascending=False).head(10)
    fig9 = px.bar(city_med, x="Price", y="City", orientation="h",
                  title="Top 10 Localities by Median Price",
                  color="Price",
                  color_continuous_scale=["#138808","#ff9900","#e94560"],
                  labels={"Price": "Median Price (Lakh)", "City": ""})
    apply_layout(fig9, height=350, coloraxis_showscale=False)
    st.plotly_chart(fig9, width="stretch")

    with st.expander("Raw Dataset (first 100 rows)"):
        st.dataframe(df.head(100), width="stretch")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — MODEL PERFORMANCE                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab3:
    if not model_ready:
        st.warning("Run `python train.py` first.")
        st.stop()

    results = payload["all_results"]
    best    = payload["model_name"]

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    m_cols = st.columns(len(results))
    for col, (mname, mvals) in zip(m_cols, results.items()):
        is_best  = mname == best
        border   = "border:1px solid #ff9900;" if is_best else ""
        title_c  = "#ff9900" if is_best else "#64748b"
        col.markdown(f"""
        <div class="card" style="{border}">
            <div style="font-size:.73rem;font-weight:700;letter-spacing:.08em;
                        text-transform:uppercase;color:{title_c};">
                {'BEST · ' if is_best else ''}{mname}
            </div>
            <div style="margin-top:.7rem;">
                <div style="font-family:Outfit;font-size:1.6rem;font-weight:700;color:#f1f5f9;">
                    {mvals['R2']:.4f}
                </div>
                <div style="font-size:.76rem;color:#64748b;">R2 Score</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.7rem;">
                <div>
                    <div style="font-size:.95rem;font-weight:600;color:#f1f5f9;">Rs.{mvals['MAE']:.1f}L</div>
                    <div style="font-size:.7rem;color:#64748b;">MAE</div>
                </div>
                <div>
                    <div style="font-size:.95rem;font-weight:600;color:#f1f5f9;">Rs.{mvals['RMSE']:.1f}L</div>
                    <div style="font-size:.7rem;color:#64748b;">RMSE</div>
                </div>
            </div>
            <div style="margin-top:.6rem;">
                <div style="font-size:.95rem;font-weight:600;color:#f1f5f9;">{mvals['CV_R2']:.4f}</div>
                <div style="font-size:.7rem;color:#64748b;">5-Fold CV R2</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar charts
    bc1, bc2 = st.columns(2, gap="medium")
    with bc1:
        r2_vals = {k: v["R2"] for k, v in results.items()}
        fig_r2  = go.Figure(go.Bar(
            x=list(r2_vals.keys()), y=list(r2_vals.values()),
            marker_color=["#ff9900" if k == best else "#64748b" for k in r2_vals],
            text=[f"{v:.4f}" for v in r2_vals.values()],
            textposition="outside", textfont=dict(color="#f1f5f9"),
        ))
        apply_layout(fig_r2, title="R2 Score Comparison", yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_r2, width="stretch")

    with bc2:
        mae_vals = {k: v["MAE"] for k, v in results.items()}
        fig_mae  = go.Figure(go.Bar(
            x=list(mae_vals.keys()), y=list(mae_vals.values()),
            marker_color=["#ff9900" if k == best else "#64748b" for k in mae_vals],
            text=[f"Rs.{v:.1f}L" for v in mae_vals.values()],
            textposition="outside", textfont=dict(color="#f1f5f9"),
        ))
        apply_layout(fig_mae, title="MAE — lower is better")
        st.plotly_chart(fig_mae, width="stretch")

    # Actual vs Predicted
    if data_ready:
        st.markdown('<div class="section-header">Actual vs Predicted</div>', unsafe_allow_html=True)
        X  = df[payload["features"]]
        y  = df["Price"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred    = payload["model"].predict(X_test)
        residuals = y_test.values - y_pred

        av1, av2 = st.columns(2, gap="medium")
        with av1:
            mn_v, mx_v = float(y_test.min()), float(y_test.max())
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(color="#ff9900", opacity=0.6, size=5), name="Predicted"))
            fig_avp.add_trace(go.Scatter(
                x=[mn_v, mx_v], y=[mn_v, mx_v], mode="lines",
                line=dict(color="#138808", dash="dash", width=2), name="Perfect Fit"))
            apply_layout(fig_avp, title="Actual vs Predicted Prices",
                         xaxis_title="Actual (Lakh)", yaxis_title="Predicted (Lakh)")
            st.plotly_chart(fig_avp, width="stretch")

        with av2:
            fig_res = px.histogram(x=residuals, nbins=50,
                                   title="Residuals Distribution",
                                   labels={"x": "Error (Lakh)"},
                                   color_discrete_sequence=["#138808"])
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ff9900", line_width=2)
            apply_layout(fig_res)
            st.plotly_chart(fig_res, width="stretch")

    # Feature importances
    try:
        reg     = payload["model"].named_steps["regressor"]
        fi      = reg.feature_importances_
        pre     = payload["model"].named_steps["preprocessor"]
        cat_enc = pre.named_transformers_["cat"]
        cat_names = cat_enc.get_feature_names_out(["Furnishing", "Floor", "City"])
        num_names = ["Area", "Bedrooms", "Bathrooms", "Parking", "Age"]
        feat_names = num_names + list(cat_names)

        fi_df = (pd.DataFrame({"Feature": feat_names, "Importance": fi})
                   .sort_values("Importance", ascending=True)
                   .tail(15))

        st.markdown('<div class="section-header">Feature Importances (Top 15)</div>', unsafe_allow_html=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=["#138808", "#ff9900", "#e94560"],
                        title=f"Feature Importances — {best}")
        apply_layout(fig_fi, coloraxis_showscale=False, height=450)
        st.plotly_chart(fig_fi, width="stretch")
    except AttributeError:
        pass


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — COMPARE PROPERTIES                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab4:
    if not model_ready:
        st.warning("Run `python train.py` first.")
        st.stop()

    st.markdown('<div class="section-header">Compare Two Indian Properties Side by Side</div>',
                unsafe_allow_html=True)

    def prop_form(suffix, color, defaults):
        area_    = st.slider(f"Area (sqft)", 350, 5500, defaults["area"], step=50, key=f"area_{suffix}")
        age_     = st.slider(f"Age (yrs)",   0,   35,   defaults["age"],           key=f"age_{suffix}")
        r1, r2, r3 = st.columns(3)
        bed_  = r1.selectbox("Bed",   [1,2,3,4,5],          index=defaults["bed"]-1,  key=f"bed_{suffix}")
        bath_ = r2.selectbox("Bath",  [1,2,3,4,5],          index=defaults["bath"]-1, key=f"bath_{suffix}")
        park_ = r3.selectbox("Park",  [0,1,2,3],            index=defaults["park"],   key=f"park_{suffix}")
        furn_ = st.selectbox("Furnishing", FURNISHING_OPTIONS, index=defaults["furn"], key=f"furn_{suffix}")
        flr_  = st.selectbox("Floor", FLOOR_OPTIONS,           index=defaults["flr"],  key=f"flr_{suffix}")
        city_ = st.selectbox("City / Locality", CITIES,
                             format_func=lambda c: f"{CITY_EMOJIS[c]} {c}",
                             index=defaults["city"], key=f"city_{suffix}")
        return area_, bed_, bath_, park_, age_, furn_, flr_, city_

    pa_col, _, pb_col = st.columns([1, 0.05, 1])

    with pa_col:
        st.markdown(f'<div style="font-family:Outfit;font-size:1.1rem;font-weight:700;color:#ff9900;margin-bottom:.6rem;">Property A</div>', unsafe_allow_html=True)
        pa = prop_form("a", "#ff9900", {"area":1200,"age":3,"bed":2,"bath":2,"park":1,"furn":1,"flr":1,"city":7})

    with pb_col:
        st.markdown(f'<div style="font-family:Outfit;font-size:1.1rem;font-weight:700;color:#138808;margin-bottom:.6rem;">Property B</div>', unsafe_allow_html=True)
        pb = prop_form("b", "#138808", {"area":1800,"age":8,"bed":3,"bath":2,"park":1,"furn":0,"flr":2,"city":12})

    st.markdown("<br>", unsafe_allow_html=True)
    compare_btn = st.button("Compare Properties", use_container_width=True)

    if compare_btn:
        pred_a = payload["model"].predict(build_input_df(*pa))[0]
        pred_b = payload["model"].predict(build_input_df(*pb))[0]
        diff   = abs(pred_a - pred_b)
        cheaper = "A" if pred_a < pred_b else "B"

        ca, mid, cb = st.columns([1, 0.25, 1])
        with ca:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,153,0,.2),rgba(255,153,0,.05));
                        border:1px solid rgba(255,153,0,.4);border-radius:16px;padding:1.5rem;text-align:center;">
                <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#ff9900;">Property A</div>
                <div style="font-family:Outfit;font-size:2rem;font-weight:800;color:#f1f5f9;margin:.5rem 0;">{format_price(pred_a)}</div>
                <div style="font-size:.78rem;color:#64748b;">{pa[0]:,} sqft · {pa[1]}BHK · {pa[7].split(' - ')[0]}</div>
            </div>""", unsafe_allow_html=True)

        with mid:
            st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:1.8rem;color:#475569;font-weight:700;">VS</div>', unsafe_allow_html=True)

        with cb:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(19,136,8,.2),rgba(19,136,8,.05));
                        border:1px solid rgba(19,136,8,.4);border-radius:16px;padding:1.5rem;text-align:center;">
                <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#138808;">Property B</div>
                <div style="font-family:Outfit;font-size:2rem;font-weight:800;color:#f1f5f9;margin:.5rem 0;">{format_price(pred_b)}</div>
                <div style="font-size:.78rem;color:#64748b;">{pb[0]:,} sqft · {pb[1]}BHK · {pb[7].split(' - ')[0]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"Property **{cheaper}** is cheaper by **{format_price(diff)}** ({diff/max(pred_a,pred_b)*100:.1f}% difference)")

        # Radar comparison — FIXED: use proper rgba hex
        FILL_A = "rgba(255,153,0,0.18)"
        FILL_B = "rgba(19,136,8,0.18)"

        def norm_vals(area, bed, bath, park, age, furn, flr, city):
            mults     = list(CITY_PRICE_MULT.values())
            norm_loc  = (CITY_PRICE_MULT[city] - min(mults)) / (max(mults) - min(mults))
            furn_s    = {"Unfurnished":0.3,"Semi-Furnished":0.6,"Fully Furnished":1.0}
            floor_s   = {"Ground":0.4,"Low (1-4)":0.6,"Mid (5-10)":0.8,"High (11+)":1.0}
            return [min(area/5500,1), bed/5, bath/5, 1-age/35, norm_loc, furn_s[furn], floor_s[flr]]

        cats = ["Area","BHK","Bathrooms","Age Factor","Location","Furnishing","Floor"]
        va, vb = norm_vals(*pa), norm_vals(*pb)

        fig_comp = go.Figure()
        for vals, name, line_c, fill_c in [
            (va, "Property A", "#ff9900", FILL_A),
            (vb, "Property B", "#138808", FILL_B),
        ]:
            fig_comp.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                name=name, fill="toself",
                fillcolor=fill_c,
                line=dict(color=line_c, width=2),
            ))
        fig_comp.update_layout(
            **PLOT_BASE,
            margin=dict(l=40, r=40, t=50, b=20),
            title="Feature Comparison Radar",
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                 tickfont=dict(size=10, color="#94a3b8")),
            ),
            height=380,
        )
        st.plotly_chart(fig_comp, width="stretch")

        # Side-by-side bar
        feat_labels = ["Area (sqft/100)","Bedrooms","Bathrooms","Parking","Price (Lakh)"]
        a_vals = [pa[0]/100, pa[1], pa[2], pa[3], pred_a]
        b_vals = [pb[0]/100, pb[1], pb[2], pb[3], pred_b]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Property A", x=feat_labels, y=a_vals, marker_color="#ff9900"))
        fig_bar.add_trace(go.Bar(name="Property B", x=feat_labels, y=b_vals, marker_color="#138808"))
        apply_layout(fig_bar, title="Side-by-Side Feature Comparison", barmode="group")
        st.plotly_chart(fig_bar, width="stretch")
