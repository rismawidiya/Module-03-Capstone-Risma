# Daegu Deals: Apartment Price Prediction

This project aims to predict apartment sale prices in **Daegu, South Korea** using machine learning techniques. By analysing various property features, the model provides estimated prices to assist real estate companies.

---

## Project Structure
- data/ # Cleaned and processed dataset
- notebooks/ # Jupyter notebooks for EDA, modeling, and evaluation
- app/ # Streamlit app for interactive prediction and analysis
- models/ # Serialised trained models and preprocessing pipeline
- requirements.txt # Python dependencies
- README.md # Project documentation

## Demo
A live demo of the project is available via Streamlit (hosted on Streamlit Cloud) and Gradio (hosted on Hugging Face):
- Run locally with: streamlit run "Capstone 03.py"
- Run locally with: python run "Capstone 03.py"

## Features Used
- HallwayType (categorical)
- TimeToSubway (ordinal)
- SubwayStation (categorical)
- N_FacilitiesNearBy(ETC)
- N_FacilitiesNearBy(PublicOffice)
- N_SchoolNearBy(University)
- N_Parkinglot(Basement)
- YearBuilt
- N_FacilitiesInApt
- Size(sqf)
- Target variable: SalePrice

## Exploratory Data Analysis (EDA)
Visualisations explored include:
- Price distribution
- Price vs individual feature
- Price vs multi-numeric features
- Price vs multi-categorical features
- Correlation heatmaps

## Models Used
All models are evaluated using cross-validation and performance metrics like RMSE, MAE, MAPE, and R² score:
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor
- HistGradient Boosting Regressor
- Ridge
- SVR

## Key Techniques
- Feature engineering
- Cross-validation
- Model tuning and evaluation
- SHAP Explainable AI for model interpretability
- Streamlit app deployment

## Evaluation
Performance of the final XGBoost model:

(After tuning)
- R² Score: 0.830756 (log)
- RMSE: 46080.200115 (₩)
- MAE: 36652.895797 (₩)
- MAPE: 18.243203 (%)

(Tried on test data)
- R² Score: 0.787331 (log)
- RMSE: 47530.272290 (₩)
- MAE: 38059.964844 (₩)
- MAPE: 18.775083 (%)
