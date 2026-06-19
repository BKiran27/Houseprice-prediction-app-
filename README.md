<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Outfit&size=36&duration=3000&pause=1000&color=FF9900&center=true&vCenter=true&width=700&lines=🏡+Ghar+Bazaar;Indian+House+Price+Prediction+AI;Powered+by+Machine+Learning" alt="Typing SVG" />

<br/>

**An end-to-end Machine Learning application that predicts Indian real estate prices across 22 city localities — from data generation to interactive web deployment.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://houseprice-prediction-app-bkiran27.streamlit.app)
&nbsp;
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BKiran27/Houseprice-prediction-app-)
&nbsp;
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://houseprice-prediction-app-bkiran27.streamlit.app)

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.5+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)

</div>

---

## 📌 Project Overview

**Ghar Bazaar** is a production-ready real estate price prediction system built for the Indian property market. It covers the complete machine learning lifecycle — from synthetic data generation and feature engineering to model training, evaluation, and deployment — all wrapped in a beautiful, interactive Streamlit web application.

The application leverages **three ML models** trained on **1,500 property records** across **22 localities in 11 major Indian cities**, providing instant price predictions, market visualizations, and side-by-side property comparisons.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🏡 Predict Price
- 8 configurable property inputs
- Instant AI-powered valuation (Lakh / Crore)
- ±8% confidence range band
- Interactive feature influence radar chart
- Price per sq ft breakdown

</td>
<td width="50%">

### 📊 Data Explorer
- Market KPI dashboard
- Price distribution histogram
- City-wise median price comparison
- BHK configuration analysis
- Furnishing & floor premium charts
- Correlation heatmap
- Age vs price trend analysis

</td>
</tr>
<tr>
<td width="50%">

### 📈 Model Performance
- 3-model benchmark cards (LR / RF / GBR)
- R², MAE, RMSE, 5-fold CV R² metrics
- Actual vs Predicted scatter plot
- Residuals distribution analysis
- Feature importances (top 15)

</td>
<td width="50%">

### 🔍 Compare Properties
- Side-by-side property configurator
- Price delta with % difference
- Feature comparison radar overlay
- Grouped bar chart comparison
- Cheaper property highlight

</td>
</tr>
</table>

---

## 🇮🇳 Indian City Coverage

> **22 localities across 11 major metros**

| Metro | Localities | Price Tier |
|-------|-----------|------------|
| 🌊 **Mumbai** | Bandra · Andheri · Thane | Premium–High |
| 🏛️ **Delhi** | South Delhi · Dwarka | Premium–High |
| 💻 **Noida / Gurgaon** | Sector 62 · DLF Phase | Mid–High |
| ☕ **Bengaluru** | Koramangala · Whitefield · Sarjapur | Premium–Mid |
| 💡 **Hyderabad** | Gachibowli · Banjara Hills | Mid–High |
| 🌳 **Pune** | Koregaon Park · Hinjewadi · Viman Nagar | Mid |
| 🌴 **Chennai** | Adyar · OMR | Mid |
| 🏫 **Kolkata** | Salt Lake · New Town | Affordable–Mid |
| 🛣️ **Ahmedabad** | SG Highway | Affordable |
| 🏰 **Jaipur** | Malviya Nagar | Affordable |
| ⛵ **Kochi** | Marine Drive | Mid |

Price range covered: **₹10 Lakh → ₹8 Crore**

---

## 🧠 Machine Learning Pipeline

```
Raw Features  ──►  Preprocessing  ──►  Model Training  ──►  Best Model  ──►  Prediction
     │                   │                    │                  │
  Numeric            StandardScaler     Linear Regression    Auto-selected      Price
  Categorical         OneHotEncoder     Random Forest        by CV R²           (Lakhs)
                                        Gradient Boosting
```

### Model Results

| Model | R² Score | MAE | RMSE | 5-Fold CV R² |
|-------|----------|-----|------|--------------|
| Linear Regression | 0.931 | Rs. 10.1 L | Rs. 13.2 L | 0.930 |
| Random Forest | 0.886 | Rs. 13.1 L | Rs. 16.9 L | 0.892 |
| **Gradient Boosting** ⭐ | **0.943** | **Rs. 9.5 L** | **Rs. 12.0 L** | **0.941** |

> The best model is **automatically selected** based on 5-fold cross-validated R² and saved as a Scikit-learn Pipeline including preprocessing.

---

## 📊 Dataset Features

| Feature | Type | Range / Values | Description |
|---------|------|----------------|-------------|
| `Area` | Numeric | 350–5,500 sqft | Super built-up area |
| `Bedrooms` | Numeric | 1–5 BHK | Number of bedrooms |
| `Bathrooms` | Numeric | 1–5 | Number of bathrooms |
| `Parking` | Numeric | 0–3 | Parking spots |
| `Age` | Numeric | 0–35 yrs | Property age |
| `Furnishing` | Categorical | Unfurnished / Semi / Fully Furnished | Furnishing status |
| `Floor` | Categorical | Ground / Low / Mid / High | Floor type |
| `City` | Categorical | 22 Indian localities | Location |
| **`Price`** | **Target** | **₹10L – ₹8Cr** | **Price in Lakhs** |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Web Framework** | Streamlit |
| **ML Library** | Scikit-learn |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Plotly Express · Graph Objects |
| **Model Storage** | Joblib |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git · GitHub |

</div>

---

## 📂 Project Structure

```
Houseprice-prediction-app-/
│
├── 📁 data/
│   └── house_data.csv          # 1,500 Indian property records (22 localities)
│
├── 📁 models/
│   └── model.pkl               # Trained pipeline — auto-generated on first run
│
├── 📄 app.py                   # Main Streamlit application (4 tabs)
├── 📄 train.py                 # ML training: 3 models + auto-selection
├── 📄 generate_data.py         # Synthetic Indian dataset generator
├── 📄 utils.py                 # City data, price formatters, helpers
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/BKiran27/Houseprice-prediction-app-.git
cd Houseprice-prediction-app-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_data.py

# 4. Train the ML models
python train.py

# 5. Launch the app
streamlit run app.py
```

> **Note:** On Streamlit Cloud, the app **automatically trains the model on first launch** — no manual steps needed.

---

## 📸 App Preview

> **4-tab interactive dashboard** with dark glassmorphism design and Indian tricolor theme

| Tab | Description |
|-----|-------------|
| **🏡 Predict** | Enter property details → Get instant AI valuation |
| **📊 Explorer** | 8+ interactive Plotly charts across all market segments |
| **📈 Performance** | Compare all 3 ML models with full diagnostics |
| **🔍 Compare** | Head-to-head property comparison with radar charts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  ┌──────────┬────────────┬───────────┬───────────┐  │
│  │ Predict  │  Explorer  │  Models   │  Compare  │  │
│  └──────────┴────────────┴───────────┴───────────┘  │
└─────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   Scikit-learn        │
              │   Pipeline            │
              │                       │
              │  StandardScaler  ──►  │
              │  OneHotEncoder   ──►  │  Gradient Boosting
              │                       │  (Best Model)
              └───────────────────────┘
                          │
              ┌───────────┴───────────┐
              │   house_data.csv      │
              │   1,500 records       │
              │   22 Indian cities    │
              └───────────────────────┘
```

---

## 💡 What I Learned

- ✅ End-to-end ML project workflow (data → model → deployment)
- ✅ Feature engineering for Indian real estate (location tiers, furnishing, floor)
- ✅ Building and comparing multiple regression models
- ✅ Scikit-learn Pipelines with ColumnTransformer for clean preprocessing
- ✅ Model evaluation: MAE, RMSE, R², 5-fold cross-validation
- ✅ Building production-ready Streamlit apps with custom CSS
- ✅ Interactive data visualization with Plotly
- ✅ Deploying ML apps to Streamlit Community Cloud

---

## 💼 Resume Description

> *Developed an end-to-end Indian real estate price prediction system using Python, Pandas, Scikit-learn, and Streamlit. Engineered a dataset of 1,500 synthetic property records across 22 localities in 11 major Indian cities (Mumbai, Delhi, Bengaluru, Hyderabad, Pune, etc.) with features including location tier, furnishing status, and floor type. Trained and benchmarked three ML models — Linear Regression, Random Forest, and Gradient Boosting — achieving a best R² of 0.943 and MAE of Rs. 9.5 Lakh. Deployed a 4-tab interactive dashboard on Streamlit Cloud featuring real-time price prediction, market analytics, model diagnostics, and property comparison tools.*

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs via [Issues](https://github.com/BKiran27/Houseprice-prediction-app-/issues)
- 💡 Suggest features or improvements
- 🔀 Submit pull requests

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

<div align="center">

**Made with ❤️ for the Indian Real Estate Market**

⭐ Star this repo if you found it useful!

[![GitHub stars](https://img.shields.io/github/stars/BKiran27/Houseprice-prediction-app-?style=social)](https://github.com/BKiran27/Houseprice-prediction-app-)

</div>
