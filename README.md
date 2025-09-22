# Retail-Customer-Segmentation-Insurance-Charges-Prediction-Crime-Analytics
# 🧠 ML Mini-Stack — Retail Segmentation (K-Means) • Insurance Regression • Crime (SVM + Forecast)

Practical, copy-run notebooks/scripts for three real-world ML tasks:

1) **Retail Customer Segmentation** — K-Means + **Elbow** + **PCA**  
2) **Insurance Charges Prediction** — Linear Regression / Random Forest / (optional) XGBoost  
3) **Crime Analytics** — **SVM** classification for crime-prone regions + **Monthly Forecast** with RF

All code is written to be **dataset-schema tolerant** (auto-detects columns when possible) and saves artifacts to `models/` and `outputs/`.

---

## 📁 Repo Layout (suggested)

.
├─ notebooks_or_scripts/
│ ├─ retail_kmeans.ipynb # or .py version with the code you pasted
│ ├─ insurance_regression.ipynb
│ └─ crime_analytics.ipynb
├─ streamlit_app.py # Insurance demo (from README)
├─ crime.py # Crime dashboard (from README)
├─ data/
│ ├─ Online Retail.xlsx # or Mall_Customers.csv
│ ├─ insurance.csv
│ └─ indian_crimes.csv
├─ models/ # saved models (auto-created)
├─ outputs/ # predictions / labeled CSVs (auto-created)
└─ README.md

yaml
Copy code

> Put your datasets in `data/` or update the paths inside the notebooks/scripts. The loaders will prompt for a path if not found.

---

## 🔧 Environment Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate

pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib streamlit

# Optional (speeds up Insurance if installed):
pip install xgboost
🧩 Task A — Retail Customer Segmentation (K-Means)
Goal: cluster customers by behavior, choose K via Elbow, visualize clusters using PCA-2D.
Dataset options (either works):

Kaggle “Mall Customers” style (columns like Age, Annual Income (k$), Spending Score (1-100)), or

UCI-style Online Retail Excel (Invoice-level) → build RFM features per CustomerID.

How it adapts automatically:

If columns contain Age / Income / Spending Score → uses them directly.

Else it assumes Online Retail schema and computes:

Recency (days since last purchase)

Frequency (distinct invoices)

Monetary (Σ Quantity × UnitPrice, filtered for returns/negatives)

Pipeline:

Load/convert .xlsx → .csv if needed

Build feature table (Age/Income/SpendingScore or RFM)

StandardScaler

Elbow Method (K = 2…10) → choose bend (default K_FINAL = 5, adjust from plot)

Fit K-Means, save labels

PCA(2) plot with cluster colors

Save:

Labeled data → outputs/task2_clusters.csv

Model + Scaler → models/task2_kmeans.joblib, models/task2_scaler.joblib

Feature meta → models/task2_feature_names.json

Datasets:

Mall Customers: https://www.kaggle.com/datasets/yasserh/mall-customers

Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail

In your script, set:

python
Copy code
DATA_PATH = "data/Online Retail.xlsx"   # or "data/Mall_Customers.csv"
💸 Task B — Insurance Charges Prediction (Regression)
Goal: predict charges from age, sex, bmi, children, smoker, region.
Models: LinearRegression, RandomForestRegressor, (optional) XGBRegressor
Metrics: MSE & R² with unified preprocessing (ColumnTransformer + OneHotEncoder + StandardScaler).

Steps:

Load data/insurance.csv

Drop junk columns like Unnamed: 0 if present

Split train/test

Train LR, RF, (XGB if installed)

Compare metrics (bar charts for MSE & R²)

Save best-by-MSE pipeline → models/insurance_best.joblib

Dataset:
https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender

🚀 Quick Streamlit UI (already provided)
Run the simple app to test your trained pipeline:

bash
Copy code
streamlit run streamlit_app.py
The app loads models/insurance_best.joblib

Enter features → click Predict charges → shows estimated amount

🚓 Task C — Crime Analytics (SVM + Monthly Forecast)
Classification: Label regions as crime-prone (top quartile of rate) vs not and train an SVM (RBF).
Forecasting: Aggregate monthly totals; train RandomForestRegressor (and a Linear baseline) to forecast.

Pipeline (robust to schema drift):

Normalize columns to lower_snake_case

Parse time (searches common date columns; requires one parseable date) → build year, month

Missing-data impact (Before) bar plot

Clean: map booleans, fix categories, light numeric imputations

Region aggregation (city/district/state) → total_crimes, crimes_per_100k (if population available)

Label crime-prone as top quartile

SVM with scaling → classification report + confusion matrix

Monthly series with lag & rolling features → train/test split → RF vs LR (MSE/R²) + line plot

Persist:

models/crime_svm.joblib

models/crime_trend_rf.joblib

models/feature_columns_trend.json

Missing-data impact (After) bar plot

Dataset (example):
https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset

🚀 Crime Streamlit Dashboard
bash
Copy code
streamlit run crime.py
Tab 1: Enter total_crimes & crimes_per_100k → SVM predicts ⚠️ Crime-prone / ✅ Safer

Tab 2: Line chart of history + Forecast using the saved RF model

If data/monthly_incidents.csv exists (columns Month, Incidents), app uses it; else, demo series

💾 Outputs & Models
Retail:

outputs/task2_clusters.csv (each customer with cluster)

models/task2_{kmeans,scaler}.joblib, models/task2_feature_names.json

Insurance:

models/insurance_best.joblib

Crime:

models/crime_svm.joblib

models/crime_trend_rf.joblib

models/feature_columns_trend.json

🧪 Quick Run Order (TL;DR)
Create env & install deps (see Environment Setup)

Retail: update DATA_PATH → run notebook/script → inspect Elbow → set K_FINAL → run PCA plot

Insurance: set INSURANCE_CSV → run cells → compare metrics → save best

Crime: set CRIME_CSV → run cells → see reports/plots → save models
