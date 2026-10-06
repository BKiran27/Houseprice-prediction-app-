"""
app.py  —  California House Price Prediction App (Streamlit)
Based on End-to-End Machine Learning Project: California Housing Prices Dataset
Features:
- Live Interactive House Price Valuation & Geographic Mapping
- Exploratory Data Analysis & Spatial Visualizations
- Multi-Model Benchmarking (Linear, Ridge, Lasso, Random Forest, Tuned HistGB)
- Residual Diagnostics & Property Comparison Engine
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    OCEAN_PROXIMITY_OPTIONS,
    OCEAN_PROXIMITY_EMOJIS,
    CALIFORNIA_PRESETS,
    format_usd,
    calculate_price_range,
    build_input_df,
    create_preprocessor
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Page Config & Custom Styling
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="California House Price AI | Machine Learning App",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.75);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.6) 50%, rgba(88, 28, 135, 0.25) 100%);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    text-align: center;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 800px;
    margin: 0 auto;
}

/* Glass Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(96, 165, 250, 0.4);
    transform: translateY(-2px);
}

/* Metric Display Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
}

.kpi-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
}

.kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* Prediction Showcase */
.prediction-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #4c1d95 100%);
    border: 1px solid rgba(129, 140, 248, 0.4);
    border-radius: 20px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 15px 45px rgba(49, 46, 129, 0.45);
}

.prediction-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255, 255, 255, 0.75);
    margin-bottom: 0.5rem;
}

.prediction-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
}

.prediction-range {
    font-size: 0.95rem;
    color: rgba(224, 231, 255, 0.85);
    margin-top: 0.6rem;
}

/* Tabs and Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.8rem;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(37, 99, 235, 0.55);
}

/* Badge Tags */
.badge {
    display: inline-block;
    background: rgba(96, 165, 250, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #93c5fd;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}

hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Plotly Theme Settings
# ═══════════════════════════════════════════════════════════════════════════════
PLOT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
    title_font=dict(family="Space Grotesk, sans-serif", color="#f8fafc", size=15),
    colorway=["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4"]
)
PLOT_MARGIN = dict(l=25, r=25, t=45, b=25)


def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOT_BASE, margin=PLOT_MARGIN, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Bootstrap & Data/Model Caching
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    model_path = os.path.join("models", "model.pkl")
    data_path = os.path.join("data", "housing.csv")

    if os.path.exists(model_path):
        return joblib.load(model_path)

    # If model is not present, train automatically on first startup
    if os.path.exists(data_path):
        from train import train_and_evaluate
        return train_and_evaluate()
    return None


@st.cache_data(show_spinner=False)
def load_dataset():
    data_path = os.path.join("data", "housing.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


payload = load_or_train_model()
df_data = load_dataset()
model_ready = payload is not None
data_ready = df_data is not None

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 0.8rem 0;'>
        <div style='font-size: 2.8rem;'>🏡</div>
        <div style='font-family: Space Grotesk; font-size: 1.45rem; font-weight: 700; color: #60a5fa;'>
            California Housing AI
        </div>
        <div style='font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;'>
            End-to-End Machine Learning System
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if model_ready:
        mn = payload["model_name"]
        test_r2 = payload["test_results"][mn]["R2"]
        test_rmse = payload["test_results"][mn]["RMSE"]

        st.markdown("**⭐ Active Primary Model**")
        st.success(mn)
        c1, c2 = st.columns(2)
        c1.metric("Test R²", f"{test_r2:.3f}")
        c2.metric("Test RMSE", f"${test_rmse/1000:.1f}k")
        st.markdown("---")
    else:
        st.warning("Model loading or training in progress...")

    st.markdown("**📚 Project Specifications**")
    st.markdown("""
    <div style='font-size: 0.82rem; color: #94a3b8; line-height: 1.8;'>
    • <b>Dataset:</b> California Housing (20,640 records)<br>
    • <b>Pipeline:</b> Median Imputer + Scaler + One-Hot<br>
    • <b>Algorithms:</b> Linear, Ridge, Lasso, RF, HistGB<br>
    • <b>Tuning:</b> 5-Fold Cross Validation + GridSearch<br>
    • <b>Primary Metric:</b> Root Mean Squared Error (RMSE)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style='font-size: 0.74rem; color: #64748b; text-align: center;'>
    Built with Scikit-learn · Streamlit · Plotly<br>
    Standard Machine Learning Project 15.4
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Hero Section
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">California House Price Prediction AI</div>
    <div class="hero-subtitle">
        An interactive machine learning application built on the California Housing Prices dataset.
        Featuring automated preprocessing pipelines, 5-fold cross-validated model selection, and hyperparameter-tuned gradient boosting.
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Application Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏡  Interactive Valuation",
    "🗺️  Geospatial & Data Explorer",
    "📈  Model Benchmarks & Residuals",
    "🔍  Property Comparison"
])

# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — INTERACTIVE VALUATION                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab1:
    if not model_ready:
        st.warning("Please wait for model training to complete.")
        st.stop()

    st.markdown("### 📍 Configure Property Features")

    # Region Preset Selector
    selected_preset_name = st.selectbox(
        "⚡ Quick-Fill with a California Location Preset:",
        list(CALIFORNIA_PRESETS.keys()),
        index=1,
        help="Choose a pre-configured California region or customize features manually."
    )

    preset_data = CALIFORNIA_PRESETS[selected_preset_name]

    col_form, col_pred = st.columns([1.15, 0.85], gap="large")

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("##### 1. Geographic Location & Coastal Proximity")
        g1, g2, g3 = st.columns([1, 1, 1.2])

        default_lon = preset_data["longitude"] if preset_data else -122.25
        default_lat = preset_data["latitude"] if preset_data else 37.85
        default_ocean = preset_data["ocean_proximity"] if preset_data else "NEAR BAY"

        with g1:
            longitude = st.number_input(
                "Longitude",
                min_value=-124.5,
                max_value=-114.0,
                value=float(default_lon),
                step=0.01,
                format="%.2f",
                help="West longitude coordinate (California range: -124.3 to -114.3)"
            )
        with g2:
            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=float(default_lat),
                step=0.01,
                format="%.2f",
                help="North latitude coordinate (California range: 32.5 to 42.0)"
            )
        with g3:
            ocean_proximity = st.selectbox(
                "Ocean Proximity",
                OCEAN_PROXIMITY_OPTIONS,
                index=OCEAN_PROXIMITY_OPTIONS.index(default_ocean),
                format_func=lambda o: OCEAN_PROXIMITY_EMOJIS.get(o, o)
            )

        st.markdown("---")
        st.markdown("##### 2. Demographics & Economic Status")
        d1, d2 = st.columns(2)

        default_income = preset_data["median_income"] if preset_data else 8.32
        default_age = preset_data["housing_median_age"] if preset_data else 41.0

        with d1:
            median_income = st.slider(
                "Block Median Income ($10,000s)",
                min_value=0.5,
                max_value=15.0,
                value=float(default_income),
                step=0.1,
                help="Measured in tens of thousands of USD. For example, 5.0 represents $50,000 median income."
            )
            st.caption(f"💵 Household income estimate: **${median_income * 10000:,.0f} / year**")

        with d2:
            housing_median_age = st.slider(
                "Median Building Age (Years)",
                min_value=1.0,
                max_value=52.0,
                value=float(default_age),
                step=1.0,
                help="Median age of structures in the block group."
            )
            st.caption(f"🏗️ Construction vintage: **approx. {int(2026 - housing_median_age)}**")

        st.markdown("---")
        st.markdown("##### 3. Block Housing Volume & Density")
        v1, v2, v3, v4 = st.columns(4)

        default_rooms = preset_data["total_rooms"] if preset_data else 2500
        default_bedrooms = preset_data["total_bedrooms"] if preset_data else 450
        default_pop = preset_data["population"] if preset_data else 1100
        default_hh = preset_data["households"] if preset_data else 420

        with v1:
            total_rooms = st.number_input("Total Rooms", min_value=10, max_value=40000, value=int(default_rooms), step=50)
        with v2:
            total_bedrooms = st.number_input("Total Bedrooms", min_value=2, max_value=8000, value=int(default_bedrooms), step=10)
        with v3:
            population = st.number_input("Block Population", min_value=5, max_value=35000, value=int(default_pop), step=25)
        with v4:
            households = st.number_input("Total Households", min_value=2, max_value=6000, value=int(default_hh), step=10)

        # Derived Ratios
        rooms_per_hh = total_rooms / max(1, households)
        bedrooms_per_room = total_bedrooms / max(1, total_rooms)
        pop_per_hh = population / max(1, households)

        st.caption(f"📊 Derived: **{rooms_per_hh:.1f}** rooms/hh · **{bedrooms_per_room:.2f}** bed ratio · **{pop_per_hh:.1f}** people/hh")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("🔮 Calculate House Price Valuation", width="stretch")

    with col_pred:
        # Run prediction
        input_row = build_input_df(
            longitude=longitude,
            latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean_proximity
        )

        model = payload["model"]
        prediction = float(model.predict(input_row)[0])
        rmse_val = payload["test_results"][payload["model_name"]]["RMSE"]
        lower_bound, upper_bound = calculate_price_range(prediction, rmse=rmse_val)

        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Median House Value</div>
            <div class="prediction-price">{format_usd(prediction)}</div>
            <div class="prediction-range">68% Confidence Interval: <b>{lower_bound} — {upper_bound}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Location mini-map using Streamlit's built-in map
        st.markdown("##### 📌 Coordinates on California Map")
        map_df = pd.DataFrame([{"lat": latitude, "lon": longitude}])
        st.map(map_df, zoom=7, width="stretch")

        # Summary Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Proximity Tier</div>
                <div class="kpi-value" style="font-size: 1.15rem;">{ocean_proximity}</div>
                <div class="kpi-sub">Coastal Zone</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Income Level</div>
                <div class="kpi-value" style="font-size: 1.15rem;">${median_income*10000:,.0f}</div>
                <div class="kpi-sub">Median Block Income</div>
            </div>
            """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — GEOSPATIAL & DATA EXPLORER                             ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab2:
    if not data_ready:
        st.warning("Dataset housing.csv is not loaded.")
        st.stop()

    st.markdown("### 🗺️ California Housing Geospatial & Exploratory Data Analysis")

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Records", f"{len(df_data):,}", "census blocks"),
        ("Median House Value", format_usd(df_data['median_house_value'].median()), "state median"),
        ("Avg Household Income", f"${df_data['median_income'].mean()*10000:,.0f}", "annual average"),
        ("Avg Building Age", f"{df_data['housing_median_age'].mean():.1f} yrs", "property vintage"),
        ("Missing Bedrooms", f"{df_data['total_bedrooms'].isna().sum()}", "imputed by pipeline")
    ]

    for col, (lbl, val, sub) in zip([k1, k2, k3, k4, k5], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{lbl}</div>
            <div class="kpi-value" style="font-size: 1.25rem;">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Geographic Scatter Map
    st.markdown("#### 🌊 Geographic Price Distribution Across California")
    # Sample 4,000 points for smooth interactive plotting
    sample_df = df_data.sample(min(4000, len(df_data)), random_state=42)

    fig_geo = px.scatter(
        sample_df,
        x="longitude",
        y="latitude",
        color="median_house_value",
        size="population",
        size_max=12,
        color_continuous_scale="Viridis",
        title="California Housing Prices by Latitude & Longitude (Population = Bubble Size)",
        labels={"median_house_value": "House Value ($)", "longitude": "Longitude", "latitude": "Latitude"},
        opacity=0.65
    )
    apply_layout(fig_geo, height=450)
    st.plotly_chart(fig_geo, width="stretch")

    # 2. Target Distribution & Ocean Proximity Boxplot
    r1, r2 = st.columns(2, gap="medium")

    with r1:
        fig_hist = px.histogram(
            df_data,
            x="median_house_value",
            nbins=50,
            title="Distribution of Median House Values (Shows $500,001 Cap)",
            color_discrete_sequence=["#3b82f6"]
        )
        fig_hist.add_vline(x=500001, line_dash="dash", line_color="#ef4444", annotation_text="Capped at $500k")
        apply_layout(fig_hist)
        st.plotly_chart(fig_hist, width="stretch")

    with r2:
        fig_box = px.box(
            df_data,
            x="ocean_proximity",
            y="median_house_value",
            color="ocean_proximity",
            title="Median House Value by Ocean Proximity",
            labels={"median_house_value": "House Value ($)", "ocean_proximity": "Ocean Proximity"}
        )
        apply_layout(fig_box, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

    # 3. Correlation Heatmap & Income vs Value
    r3, r4 = st.columns(2, gap="medium")

    with r3:
        corr_matrix = df_data.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix"
        )
        apply_layout(fig_corr)
        st.plotly_chart(fig_corr, width="stretch")

    with r4:
        fig_income = px.scatter(
            sample_df,
            x="median_income",
            y="median_house_value",
            color="ocean_proximity",
            opacity=0.6,
            title="Median Income vs House Value (Strongest Predictor: r = 0.69)",
            labels={"median_income": "Median Income ($10k)", "median_house_value": "House Value ($)"}
        )
        apply_layout(fig_income)
        st.plotly_chart(fig_income, width="stretch")

    # Raw Data Explorer
    with st.expander("🗃️ View Raw California Housing Dataset (First 100 Rows)"):
        st.dataframe(df_data.head(100), width="stretch")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — MODEL BENCHMARKS & RESIDUALS                           ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab3:
    if not model_ready:
        st.warning("Model benchmark statistics not ready.")
        st.stop()

    cv_results = payload["cv_results"]
    test_results = payload["test_results"]
    best_name = payload["model_name"]

    st.markdown("### 📊 Model Selection & Performance Benchmarking")
    st.markdown(
        "Following the methodology in Project 15.4: 5 algorithms evaluated via **5-fold Cross-Validation** "
        "on the training set, followed by **GridSearchCV hyperparameter tuning** on the best performing model."
    )

    # Benchmark Cards
    b_cols = st.columns(len(cv_results))
    for col, (m_name, res) in zip(b_cols, test_results.items()):
        is_best = (m_name == best_name)
        border_style = "border: 1px solid #3b82f6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.25);" if is_best else "border: 1px solid rgba(255, 255, 255, 0.08);"
        title_color = "#60a5fa" if is_best else "#94a3b8"

        col.markdown(f"""
        <div class="glass-card" style="{border_style}; min-height: 210px;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {title_color};">
                {'🏆 ' if is_best else ''}{m_name}
            </div>
            <div style="font-family: Space Grotesk; font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin: 0.5rem 0;">
                {res['R2']:.4f}
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: -0.4rem;">Test R² Score</div>
            <hr style="margin: 0.6rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span style="color: #94a3b8;">Test RMSE:</span>
                <span style="color: #f8fafc; font-weight: 600;">${res['RMSE']:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.2rem;">
                <span style="color: #94a3b8;">Test MAE:</span>
                <span style="color: #f8fafc; font-weight: 600;">${res['MAE']:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.2rem;">
                <span style="color: #94a3b8;">CV RMSE:</span>
                <span style="color: #a78bfa; font-weight: 600;">${cv_results[m_name]['RMSE']:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Bar Charts
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        # RMSE Comparison (Lower is Better)
        rmse_df = pd.DataFrame([
            {"Model": name, "Test RMSE": res["RMSE"], "Best": (name == best_name)}
            for name, res in test_results.items()
        ]).sort_values("Test RMSE", ascending=False)

        fig_rmse = px.bar(
            rmse_df,
            x="Test RMSE",
            y="Model",
            orientation="h",
            color="Test RMSE",
            color_continuous_scale="Viridis_r",
            title="Root Mean Squared Error (RMSE) — Lower is Better",
            labels={"Test RMSE": "Test RMSE ($)", "Model": ""}
        )
        apply_layout(fig_rmse, coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, width="stretch")

    with c2:
        # R2 Comparison (Higher is Better)
        r2_df = pd.DataFrame([
            {"Model": name, "Test R²": res["R2"], "Best": (name == best_name)}
            for name, res in test_results.items()
        ]).sort_values("Test R²", ascending=True)

        fig_r2 = px.bar(
            r2_df,
            x="Test R²",
            y="Model",
            orientation="h",
            color="Test R²",
            color_continuous_scale="Purples",
            title="Coefficient of Determination (R²) — Higher is Better",
            labels={"Test R²": "Test R² Score", "Model": ""}
        )
        apply_layout(fig_r2, coloraxis_showscale=False)
        st.plotly_chart(fig_r2, width="stretch")

    # Residuals & Actual vs Predicted Analysis
    st.markdown("#### 🔬 Residual Error Analysis & Actual vs Predicted")
    d1, d2 = st.columns(2, gap="medium")

    sample_y = payload["sample_test_y"]
    sample_pred = payload["sample_test_pred"]
    residuals = payload["residuals"]

    with d1:
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=sample_y,
            y=sample_pred,
            mode="markers",
            marker=dict(color="#3b82f6", opacity=0.5, size=5),
            name="Predictions"
        ))
        min_v = float(min(sample_y.min(), sample_pred.min()))
        max_v = float(max(sample_y.max(), sample_pred.max()))
        fig_avp.add_trace(go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            line=dict(color="#f43f5e", dash="dash", width=2),
            name="Perfect Fit (y = x)"
        ))
        apply_layout(
            fig_avp,
            title="Actual vs Predicted Values (Test Set)",
            xaxis_title="Actual House Value ($)",
            yaxis_title="Predicted Value ($)"
        )
        st.plotly_chart(fig_avp, width="stretch")

    with d2:
        fig_res = px.histogram(
            x=residuals,
            nbins=50,
            title="Residual Error Distribution (Actual - Predicted)",
            labels={"x": "Prediction Error ($)"},
            color_discrete_sequence=["#8b5cf6"]
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#f43f5e", line_width=2)
        apply_layout(fig_res)
        st.plotly_chart(fig_res, width="stretch")

    # Hyperparameter Tuning Details
    with st.expander("⚙️ View GridSearchCV Optimal Hyperparameters for HistGradientBoosting"):
        st.json(payload["best_params"])

# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — PROPERTY COMPARISON                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab4:
    if not model_ready:
        st.warning("Model not ready.")
        st.stop()

    st.markdown("### 🔍 Compare Two California Locations Side by Side")

    def render_prop_inputs(suffix: str, default_preset_idx: int):
        preset_choice = st.selectbox(
            f"Preset for Property {suffix}:",
            list(CALIFORNIA_PRESETS.keys()),
            index=default_preset_idx,
            key=f"preset_{suffix}"
        )
        p_data = CALIFORNIA_PRESETS[preset_choice]

        c1, c2 = st.columns(2)
        with c1:
            lon = st.number_input("Longitude", -124.5, -114.0, float(p_data["longitude"] if p_data else -122.2), step=0.01, key=f"lon_{suffix}")
            inc = st.slider("Median Income ($10k)", 0.5, 15.0, float(p_data["median_income"] if p_data else 6.0), step=0.1, key=f"inc_{suffix}")
            rooms = st.number_input("Total Rooms", 50, 20000, int(p_data["total_rooms"] if p_data else 2500), step=50, key=f"rooms_{suffix}")
            pop = st.number_input("Population", 10, 15000, int(p_data["population"] if p_data else 1200), step=25, key=f"pop_{suffix}")
        with c2:
            lat = st.number_input("Latitude", 32.0, 42.0, float(p_data["latitude"] if p_data else 37.8), step=0.01, key=f"lat_{suffix}")
            age = st.slider("Building Age", 1.0, 52.0, float(p_data["housing_median_age"] if p_data else 30.0), step=1.0, key=f"age_{suffix}")
            beds = st.number_input("Total Bedrooms", 10, 5000, int(p_data["total_bedrooms"] if p_data else 500), step=10, key=f"beds_{suffix}")
            hh = st.number_input("Households", 5, 4000, int(p_data["households"] if p_data else 450), step=10, key=f"hh_{suffix}")

        ocean = st.selectbox(
            "Ocean Proximity",
            OCEAN_PROXIMITY_OPTIONS,
            index=OCEAN_PROXIMITY_OPTIONS.index(p_data["ocean_proximity"] if p_data else "<1H OCEAN"),
            key=f"ocean_{suffix}"
        )
        return lon, lat, age, rooms, beds, pop, hh, inc, ocean

    cp1, cp2 = st.columns(2, gap="large")
    with cp1:
        st.markdown("#### 🅰️ Property A")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        prop_a = render_prop_inputs("A", 1) # SF preset
        st.markdown('</div>', unsafe_allow_html=True)

    with cp2:
        st.markdown("#### 🅱️ Property B")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        prop_b = render_prop_inputs("B", 5) # Sacramento preset
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    comp_btn = st.button("⚖️ Run Comparative Valuation", width="stretch")

    if comp_btn or "last_comp" in st.session_state:
        df_a = build_input_df(*prop_a)
        df_b = build_input_df(*prop_b)

        val_a = float(payload["model"].predict(df_a)[0])
        val_b = float(payload["model"].predict(df_b)[0])
        diff = abs(val_a - val_b)

        st.session_state["last_comp"] = True

        r_a, r_mid, r_b = st.columns([1, 0.4, 1])
        with r_a:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-color: #3b82f6;">
                <div style="color: #60a5fa; font-weight: 700; font-size: 0.85rem;">PROPERTY A VALUATION</div>
                <div style="font-family: Space Grotesk; font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0.4rem 0;">
                    {format_usd(val_a)}
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem;">{prop_a[8]} · Inc: ${prop_a[7]*10000:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r_mid:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 800; color: #a78bfa;">VS</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.4rem;">Delta: <b>{format_usd(diff)}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with r_b:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-color: #a78bfa;">
                <div style="color: #c084fc; font-weight: 700; font-size: 0.85rem;">PROPERTY B VALUATION</div>
                <div style="font-family: Space Grotesk; font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0.4rem 0;">
                    {format_usd(val_b)}
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem;">{prop_b[8]} · Inc: ${prop_b[7]*10000:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparative Feature Bar Chart
        comp_metrics = ["Estimated Price ($10k)", "Median Income ($10k)", "Rooms / 100", "Age (Yrs)"]
        vals_a = [val_a / 10000, prop_a[7], prop_a[3] / 100, prop_a[2]]
        vals_b = [val_b / 10000, prop_b[7], prop_b[3] / 100, prop_b[2]]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Property A", x=comp_metrics, y=vals_a, marker_color="#3b82f6"))
        fig_comp.add_trace(go.Bar(name="Property B", x=comp_metrics, y=vals_b, marker_color="#a78bfa"))
        apply_layout(fig_comp, title="Side-by-Side Feature & Valuation Comparison", barmode="group")
        st.plotly_chart(fig_comp, width="stretch")
