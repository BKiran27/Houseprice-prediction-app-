# 🏡 Ghar Bazaar — Indian House Price Prediction AI

> An end-to-end Machine Learning application that predicts real estate prices across **22 Indian city localities** using Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://houseprice-prediction-app-bkiran27.streamlit.app)

### 🔗 [Live Demo → https://houseprice-prediction-app-bkiran27.streamlit.app](https://houseprice-prediction-app-bkiran27.streamlit.app)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ghar-bazaar.git
cd ghar-bazaar

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_data.py

# 4. Train the ML models
python train.py

# 5. Launch the Streamlit app
streamlit run app.py
```

App opens at → **http://localhost:8501**

---

## 🇮🇳 Indian City Localities Covered

| Metro | Localities |
|-------|-----------|
| 🌊 **Mumbai** | Bandra, Andheri, Thane |
| 🏛️ **Delhi** | South Delhi, Dwarka |
| 💻 **Noida / Gurgaon** | Sector 62, DLF Phase |
| ☕ **Bengaluru** | Koramangala, Whitefield, Sarjapur |
| 💡 **Hyderabad** | Gachibowli, Banjara Hills |
| 🌳 **Pune** | Koregaon Park, Hinjewadi, Viman Nagar |
| 🌴 **Chennai** | Adyar, OMR |
| 🏫 **Kolkata** | Salt Lake, New Town |
| 🛣️ **Ahmedabad** | SG Highway |
| 🏰 **Jaipur** | Malviya Nagar |
| ⛵ **Kochi** | Marine Drive |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Area | Numeric | Super built-up area (350–5500 sq ft) |
| Bedrooms | Numeric | 1–5 BHK |
| Bathrooms | Numeric | 1–5 |
| Parking | Numeric | 0–3 spots |
| Age | Numeric | Property age in years (0–35) |
| Furnishing | Categorical | Unfurnished / Semi-Furnished / Fully Furnished |
| Floor | Categorical | Ground / Low (1-4) / Mid (5-10) / High (11+) |
| City | Categorical | 22 Indian city localities |
| **Price** | **Target** | **Price in Lakhs (₹)** |

---

## 🧠 ML Models

| Model | Description |
|-------|-------------|
| Linear Regression | Interpretable baseline |
| Random Forest | 200 decision trees ensemble |
| **Gradient Boosting** | Sequential boosting — best performer |

The best model is auto-selected by 5-fold cross-validated R² and saved to `models/model.pkl`.

---

## 🎨 App Features

| Tab | Features |
|-----|---------|
| 🏡 **Predict Price** | 8 input controls, animated price box, confidence range, feature radar chart |
| 📊 **Data Explorer** | 8+ interactive charts — price distribution, city medians, BHK analysis, furnishing impact |
| 📈 **Model Performance** | 3-model comparison, Actual vs Predicted, Residuals, Feature Importances |
| 🔍 **Compare Properties** | Side-by-side configurator, radar overlay, grouped bar chart |

---

## 📂 Project Structure

```
ghar-bazaar/
├── data/
│   └── house_data.csv          # 1,500 synthetic Indian property records
├── models/
│   └── model.pkl               # Best trained pipeline (auto-generated)
├── app.py                      # Streamlit app (4 tabs)
├── train.py                    # Model training & evaluation
├── generate_data.py            # Synthetic dataset generator
├── utils.py                    # City data, formatters, helpers
├── requirements.txt
└── README.md
```

---

## 📈 Expected Model Performance

| Model | R² | MAE | CV R² |
|-------|----|-----|-------|
| Linear Regression | ~0.91 | ~7L | ~0.93 |
| Random Forest | ~0.96 | ~5L | ~0.96 |
| **Gradient Boosting** | **~0.97** | **~4.5L** | **~0.97** |

---

## 🛠️ Tech Stack

- **Streamlit** — Interactive web UI
- **Scikit-learn** — ML pipelines (preprocessing + models)
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing
- **Joblib** — Model serialization

---

## 💼 Resume Description

> Built an end-to-end Indian real estate price prediction system using Python, Scikit-learn, and Streamlit. Modeled property pricing across 22 city localities including Mumbai, Delhi, Bengaluru, and Hyderabad. Engineered features for furnishing status, floor level, and location tier. Trained and compared Linear Regression, Random Forest, and Gradient Boosting achieving R² of 0.97. Deployed a 4-tab interactive Streamlit dashboard with live prediction, data visualization, model benchmarking, and property comparison.

---

## 📄 License

MIT License — free to use, modify, and distribute.
