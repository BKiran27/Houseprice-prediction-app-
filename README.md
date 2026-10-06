<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&size=34&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=750&lines=🏠+California+House+Price+Prediction;End-to-End+Machine+Learning+System;Cross-Validated+Pipeline+%2B+Interactive+App" alt="Typing SVG" />

<br/>

**A production-grade Machine Learning system for predicting housing values based on the California Housing Prices dataset — featuring automated preprocessing pipelines, 5-fold cross-validated model selection, GridSearchCV hyperparameter tuning, and an interactive Streamlit web dashboard.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://houseprice-prediction-app-bkiran27.streamlit.app)
&nbsp;
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BKiran27/Houseprice-prediction-app-)
&nbsp;
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://houseprice-prediction-app-bkiran27.streamlit.app)

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square)

</div>

---

## 📌 Project Overview

This project is a complete implementation of the **California House Price Prediction** end-to-end machine learning project. The goal is to estimate the `median_house_value` for California census block groups using demographic, geographic, and housing features.

### 🔄 End-to-End ML Lifecycle
```
Data Ingestion (20,640 records)
       │
       ▼
Exploratory Data Analysis (EDA & Geospatial Distributions)
       │
       ▼
Preprocessing Pipeline (Median Imputer + StandardScaler + OneHotEncoder)
       │
       ▼
Model Selection (5-Fold Cross Validation across 5 Regressors)
       │
       ▼
Hyperparameter Tuning (GridSearchCV on HistGradientBoosting)
       │
       ▼
Final Model Evaluation & Residual Diagnostics (RMSE, MAE, R²)
       │
       ▼
Deployment (Interactive Streamlit Dashboard with Geospatial Mapping)
```

---

## 🏆 Model Benchmarking & Performance

All models were evaluated using **5-Fold Cross Validation** on the training set and assessed on a held-out test set (80/20 split, random_state=42):

| Model Architecture | 5-Fold CV RMSE | Test RMSE | Test MAE | Test R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | $68,604 | $70,059 | $50,670 | 0.6254 | Baseline |
| **Ridge Regression** | $68,595 | $70,066 | $50,676 | 0.6254 | Regularized L2 |
| **Lasso Regression** | $68,603 | $70,060 | $50,671 | 0.6254 | Regularized L1 |
| **Random Forest Regressor** | $49,445 | $48,941 | $31,628 | 0.8172 | Ensemble |
| **HistGradientBoosting** (Default) | $48,250 | $48,106 | $32,456 | 0.8234 | Gradient Boosted |
| **Tuned HistGradientBoosting** ⭐ | **$47,472** | **$46,610** | **$30,878** | **0.8342** | **Best Model** |

> **Key Takeaway:** Hyperparameter tuning via GridSearchCV reduced test RMSE from **$70,059** (baseline Linear Regression) down to **$46,610** while explaining over **83.4%** of the variance in California house values.

---

## 📊 Dataset & Feature Dictionary

The model is trained on **20,640 census block records** from the 1990 California Census:

| Feature | Type | Description |
| :--- | :---: | :--- |
| `longitude` | Numeric | Geographic longitude coordinate (West is negative) |
| `latitude` | Numeric | Geographic latitude coordinate |
| `housing_median_age` | Numeric | Median building age within the block (1 to 52 years) |
| `total_rooms` | Numeric | Total number of rooms in all housing units |
| `total_bedrooms` | Numeric | Total bedrooms (contains 207 missing values handled by Imputer) |
| `population` | Numeric | Total resident population in the block |
| `households` | Numeric | Total household units in the block |
| `median_income` | Numeric | Median household income (measured in tens of thousands of USD) |
| `ocean_proximity` | Categorical | Location relative to the Pacific Ocean (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`) |
| **`median_house_value`** | **Target** | **Median house value in USD (Capped at $500,001)** |

---

## ✨ Web Application Features

The interactive web dashboard is built with **Streamlit** and **Plotly**:

### 1. 🏡 Interactive Valuation Engine
- **Preset Selector:** One-click auto-fill for famous California regions (Bay Area / San Francisco, Silicon Valley, Santa Monica, San Diego Coastal, Sacramento Suburbs, Central Valley, Catalina Island).
- **Interactive Location Map:** Visualizes coordinates directly on a live California map.
- **Instant Inference:** Calculates estimated house value with ±RMSE confidence bounds.
- **Derived Metrics:** Automatically computes rooms per household, bedroom ratios, and occupancy density.

### 2. 🗺️ Geospatial & Exploratory Data Analysis
- **Geographic Scatter Map:** Visualizes 4,000+ census blocks colored by price and sized by population.
- **Target Distribution:** Histogram displaying the distribution and the $500k ceiling effect.
- **Coastal Impact:** Boxplot analysis highlighting coastal vs inland price premiums.
- **Correlation Heatmap:** Demonstrates the strong correlation ($r = 0.69$) between `median_income` and housing price.

### 3. 📈 Model Diagnostics & Residuals
- **Model Benchmark Cards:** Side-by-side comparison across all 5 models.
- **Actual vs Predicted Plot:** Scatter evaluation against the perfect-fit diagonal ($y = x$).
- **Residual Distribution:** Visualizes prediction errors to verify homoscedasticity.
- **Optimal Hyperparameters:** Displays GridSearchCV settings.

### 4. 🔍 Dual Property Comparison
- Allows side-by-side configuration of two properties / locations to compute price differentials and comparative feature metrics.

---

## 📂 Repository Structure

```
Houseprice-prediction-app-/
│
├── 📁 data/
│   └── housing.csv                 # Complete California Housing dataset (20,640 rows)
│
├── 📁 models/
│   └── model.pkl                   # Serialized tuned pipeline & evaluation metrics
│
├── 📁 notebooks/
│   └── house_price_prediction.ipynb # Complete step-by-step Jupyter Notebook
│
├── 📁 .streamlit/
│   └── config.toml                 # Custom dark theme configuration
│
├── 📄 app.py                       # Full-featured Streamlit web application (4 tabs)
├── 📄 train.py                     # ML pipeline: Preprocessing, CV, tuning, evaluation
├── 📄 utils.py                     # Feature helpers, presets, and price formatters
├── 📄 requirements.txt             # Python package dependencies
├── 📄 LICENSE                      # MIT Open-Source License
├── 📄 .gitignore
└── 📄 README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/BKiran27/Houseprice-prediction-app-.git
cd Houseprice-prediction-app-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-Train Models
```bash
python train.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```

---

## 💼 Resume & Portfolio Summary

> **California Real Estate Price Prediction System (Machine Learning & Web App)**
> *Developed an end-to-end Machine Learning system using Python, Scikit-learn, Pandas, and Streamlit to predict residential property values across California census block groups. Engineered automated ColumnTransformer pipelines with median imputation, feature scaling, and one-hot encoding. Conducted 5-fold cross-validation across 5 regression models (Linear, Ridge, Lasso, Random Forest, HistGradientBoosting) and optimized hyperparameter combinations using GridSearchCV, achieving a test RMSE of $46,610 and R² of 0.834. Deployed an interactive 4-tab Streamlit application with geospatial mapping, comparative property valuation, and residual error diagnostics.*

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and adapt it for learning and portfolio purposes.
