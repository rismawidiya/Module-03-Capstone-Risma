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

## Conclusions:

After a thorough comparison of multiple regression models, from ensemble methods like Random Forest and Gradient Boosting to regularised linear models and SVR, **XGBoost** emerged as the most consistent and well-rounded performer. While the improvement after tuning was modest, **XGBoost achieved the lowest MAE (₩38,060) and RMSE (₩47,530)** on the test set, alongside the **highest R² score (0.787)** in predicting log-transformed sale prices. It also maintained a **balanced MAPE (18.78%)**, confirming its reliability across price ranges.

1. Model Performance Insights:
    - All ensemble tree-based models outperformed Ridge and SVR in terms of R² and error metrics.
    - XGBoost slightly edged out Gradient Boosting and CatBoost after tuning, showing it captured feature interactions more effectively.
    - Compared to a simple baseline using Grouped Mean by Size, XGBoost reduced MAE by over 55% and massively improved R² from -0.02 to 0.79, proving the advantage of using a multi-feature ML model.

2. Error & Residual Patterns:
    - Residuals were fairly well-distributed, with no major sign of systematic bias though predictions became less precise in higher price brackets (> ₩250k).
    - Most predictions were classified as accurate, but underpricing tended to appear more in the ₩100k–₩300k range. Overpricing was more spread out, especially in mid-range prices.

3. Model Interpretability (SHAP):
    - Size(sqf) was consistently the most influential feature, with newer buildings (YearBuilt) and Terraced hallways also playing major roles in boosting prices.
    - Basement parking, proximity to ETC facilities, and subway access had more localised or interaction-based influence, sometimes increasing or slightly reducing predictions depending on context.
    - The model captured nuanced effects. For instance, larger size didn’t always mean higher price, especially when combined with certain other features like parking or excess amenities.

4. Prediction Examples:
    - Smaller, older apartments with less favourable layouts were predicted below average, even when they had some premium features.
    - On the flip side, larger, newer units with desirable layouts (Terraced) were consistently priced higher by the model, though some features like basement parking occasionally acted as mild price suppressors depending on combinations.

## Recommendations:

1. **Apply the model cautiously to high-end listings**:

    For premium properties priced above ₩250,000, model accuracy decreases due to greater price variability.

    *Action:* Combine the model’s estimate with expert appraisals and enhanced listing data (e.g., floor level, view quality, interior finish level) to improve confidence and pricing precision.

2. **Highlight premium traits in marketing**:

    Larger apartments, newer buildings, and those with terraced hallway types command higher prices.

    *Action:* Highlight these features explicitly in marketing materials and filter priority properties for promotion using the model’s top feature drivers.

3. **Use model explanations for buyer-seller transparency**:

    SHAP explanations (e.g., for Index 25 and 35) show which features drive pricing and why. This builds credibility with clients.

    *Action:* Integrate these visual explanations into client presentations or team dashboards to justify prices and support negotiations with data.

4. **Consider adding neighbourhood-level features**:

    External factors like school quality, crime rates, or urban development explain variance that internal features alone cannot.

    *Action:* Partner with local data providers or open government sources to enrich listings with these features. Pilot this in high-variance zones first to evaluate impact on pricing accuracy.

---

- Presentation: [Daegu Apartments Presentation](https://drive.google.com/file/d/1WVzdPRpZZJxKvqw2AgML9rBdZCChfKlA/view?usp=sharing)
- Streamlit: [Daegu Apartments Price Predictor](https://module-03-capstone-risma-daegu-apartments.streamlit.app)
- Gradio: [Daegu Apartments Price Predictor](https://rismawidiya-portfolio-project.hf.space)
