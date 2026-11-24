# ⚡ TRANSFORMER IQ
## AI-Based Transformer Oil Temperature Forecasting System

TRANSFORMER IQ is a Machine Learning-based application designed to forecast transformer oil temperature 24 hours ahead using historical time-series data.

The project combines Electrical Engineering concepts, Machine Learning, time-series analysis, and an interactive Streamlit dashboard.

> **Project Status:** Research Prototype / Development

---

## 📌 1. Project Overview

Transformers generate heat during operation due to electrical losses, including copper losses and core losses.

Monitoring oil temperature is important for understanding transformer thermal behaviour.

The objective of this project is to use historical temperature and load-related data to predict oil temperature 24 hours in advance.

The system also provides interactive visualizations and model performance analysis through a Streamlit dashboard.

## 🎯 2. Project Objectives

- Analyze historical transformer temperature data.
- Understand relationships between load-related features and oil temperature.
- Perform exploratory data analysis (EDA).
- Create time-series features using historical observations.
- Train and evaluate Machine Learning regression models.
- Forecast oil temperature 24 hours ahead.
- Develop an interactive industrial-style dashboard.

---

## 📊 3. Dataset Information

**Dataset:** Electricity Transformer Temperature (ETT)

**File used:** ETTh1.csv

**Data frequency:** Hourly

The dataset contains the following columns:

| Column | Description |
|---|---|
| date | Date and timestamp |
| HUFL | High Useful Load |
| HULL | High Useless Load |
| MUFL | Middle Useful Load |
| MULL | Middle Useless Load |
| LUFL | Low Useful Load |
| LULL | Low Useless Load |
| OT | Oil Temperature |

`OT` is the target variable used for forecasting.

The six load-related features are used as input variables.

Dataset source: https://github.com/zhouhaoyi/ETDataset

---

## 🔍 4. Exploratory Data Analysis (EDA)

The following analysis was performed:

- Dataset inspection
- Missing value analysis
- Duplicate record checking
- Statistical summary
- Oil temperature visualization
- Load feature visualization
- Correlation matrix analysis

### Correlation Analysis

The correlation analysis showed:

- HUFL and MUFL have a strong positive correlation.
- HULL and MULL also have a strong positive correlation.
- The contemporaneous correlations between individual load-related features and OT were relatively weak.

This indicates that oil temperature forecasting may require consideration of historical temperature patterns and thermal behaviour.

---

## ⚙️ 5. Feature Engineering

Time-series features were created to help Machine Learning models learn historical patterns.

### Oil Temperature Lag Features

| Feature | Description |
|---|---|
| OT_lag_1 | Oil temperature 1 hour earlier |
| OT_lag_6 | Oil temperature 6 hours earlier |
| OT_lag_12 | Oil temperature 12 hours earlier |
| OT_lag_24 | Oil temperature 24 hours earlier |
| OT_lag_48 | Oil temperature 48 hours earlier |

### Load Lag Features

24-hour lag features were created for:

- HUFL
- HULL
- MUFL
- MULL
- LUFL
- LULL

### Prediction Target

The target variable was created as:

`OT_future_24h`

This represents the oil temperature 24 hours after the current observation.

The dataset was divided chronologically:

- 80% Training data
- 20% Testing data

Random shuffling was avoided to preserve time-series order.

---

## 🤖 6. Machine Learning Models

The following approaches were experimented with:

### 1. Baseline Model

A persistence forecasting approach that uses current oil temperature as the prediction for future oil temperature.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to perform regression.

### 3. Gradient Boosting Regressor

An ensemble algorithm that builds trees sequentially to improve prediction performance.

### 4. Tuned Random Forest

RandomizedSearchCV was used to tune Random Forest hyperparameters.

TimeSeriesSplit was used for chronological cross-validation.

### 5. Thermal Feature Experiment

Additional features were tested:

- Rolling mean of oil temperature
- Rolling standard deviation
- Temperature changes
- Rolling load averages

The experiment did not improve the Random Forest performance.

---

## 📈 7. Model Evaluation

Models were evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² Score — Coefficient of Determination

### Fair Test-Set Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 1.714700 | 2.239401 | 0.577118 |
| Tuned Random Forest | 1.715222 | 2.244271 | 0.575277 |

### Key Findings

- The Baseline model currently performs slightly better than the tuned Random Forest.
- Random Forest did not achieve a meaningful improvement over the Baseline.
- Removing the current OT feature reduced Random Forest performance.
- Additional thermal features also reduced performance in the tested configuration.

These results demonstrate why comparing Machine Learning models against simple baselines is important.

**Note:** Earlier Gradient Boosting and other experiments were conducted in separate experiment stages. Their results should not be treated as a final same-test-set comparison.

---

## 🖥️ 8. Streamlit Dashboard

An interactive industrial-style dashboard was developed using Streamlit and Plotly.

### Dashboard Modules

| Module | Functionality |
|---|---|
| Command Center | Transformer overview and forecast summary |
| Thermal Analytics | Historical oil temperature visualization |
| Load Intelligence | Load-related feature analysis |
| AI Forecasting | 24-hour oil temperature prediction |
| Model Lab | Model performance comparison |
| Data Explorer | CSV inspection and processed data download |

### Dashboard Features

- Dark industrial user interface
- CSV file upload
- Automatic lag feature generation
- 24-hour oil temperature prediction
- Interactive Plotly charts
- Model performance metrics
- Historical temperature analysis

---

## 🛠️ 9. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| Scikit-learn | Machine Learning |
| Random Forest | Regression model |
| Gradient Boosting | Regression model |
| Joblib | Model serialization |
| Streamlit | Web application |

---

## 📁 10. Project Structure

```text
Transformer_AI_Project/
│
├── data/
│   └── ETTh1.csv
│
├── models/
│   ├── model_info.pkl
│   └── transformer_random_forest.pkl
│
├── notebooks/
│   └── Transformer_Analysis.ipynb
│
├── reports/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 11. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open Project Folder

```bash
cd Transformer_AI_Project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Dataset

Place `ETTh1.csv` inside the `data/` folder.

### Step 5: Run Streamlit

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

---

## 📂 12. CSV Input Requirements

The current dashboard expects CSV files containing:

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

Requirements:

- At least 49 valid hourly records.
- Chronologically ordered timestamps.
- Compatible feature definitions and units.
- No missing values in required input columns.

The application automatically creates the required lag features.

The current model is trained on the ETTh1 dataset. Different transformer sensor datasets require appropriate feature mapping and model validation or retraining before reliable use.

---

## 🔮 13. Future Improvements

Planned development:

- AI-based anomaly detection
- Improved time-series forecasting
- Better model performance
- Explainable AI and feature analysis
- Custom practice dataset generator
- Real transformer sensor integration
- IoT-based monitoring
- Advanced temperature trend visualization
- Deployment to a public cloud platform

---

## ⚠️ 14. Limitations and Disclaimer

TRANSFORMER IQ is currently a research and educational forecasting prototype.

- It forecasts the dataset's oil temperature target.
- It does not diagnose transformer faults.
- It is not a certified transformer protection system.
- Predictions depend on the compatibility and quality of uploaded data.
- The current model should not be used for real-world operational safety decisions.

---

## 👨‍💻 15. Project Purpose

This project was developed to explore the integration of Electrical Engineering and Machine Learning.

It demonstrates practical experience with:

- Transformer-related data analysis
- Time-series feature engineering
- Regression modelling
- Model evaluation and experimentation
- Interactive dashboard development

**TRANSFORMER IQ — Bringing Electrical Engineering and Artificial Intelligence Together.** ⚡

---

## 📌 Acknowledgements

Dataset: Electricity Transformer Temperature (ETT)

Repository: https://github.com/zhouhaoyi/ETDataset