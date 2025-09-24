# Daegu Deals: What's the Real Price of Home Apartment?

## Project Overview  
The Daegu real estate market is dynamic, with apartment prices influenced by factors such as location, building age, size, and access to public transport. Accurately estimating property values is critical for **real estate companies, buyers, and sellers** to make informed decisions in a competitive housing market.  

This project applies machine learning to analyse apartment features and predict sale prices, demonstrating how a data-driven approach can improve **pricing strategies, marketing, and client trust**. Beyond prediction, the project uses explainable AI to uncover which features most impact property values, offering actionable insights for the industry.  

Key focus areas include:  
- **Price Distribution Analysis:** How are apartment prices distributed across Daegu?  
- **Feature Relationships:** Which property attributes (size, year built, hallway type, subway access) most strongly influence price?  
- **Model Benchmarking:** How do different models (Random Forest, Gradient Boosting, XGBoost, etc.) compare in predictive accuracy?  
- **Residual & Error Patterns:** Are certain price ranges harder to predict accurately?  
- **SHAP Interpretability:** Which features consistently drive prices up or down?  
- **Business Recommendations:** How can model insights support pricing, negotiations, and marketing strategies?

---

## Executive Summary  
This project analyses **Daegu apartment sales data** to identify the key drivers of property prices and build predictive models for more accurate valuations. After cleaning and preparing the dataset, I explored how property features such as **size, building age, hallway type, and subway proximity** affect price.  

Key outcomes include:  
- **Apartment size** as the strongest driver of price, with newer buildings and terraced hallways further boosting value  
- **Tree-based ensemble models** (Random Forest, Gradient Boosting, XGBoost, CatBoost) significantly outperforming linear models  
- **XGBoost** emerging as the most reliable model with an R² of 0.79 and MAE of ₩38,060 on test data  
- **Residual patterns** showing stronger accuracy in mid-range properties, with slightly reduced precision for high-end listings (> ₩250,000)  
- **SHAP analysis** confirming nuanced interactions: size doesn’t always guarantee higher prices when combined with certain features like basement parking or excess amenities  

These findings translate into **actionable recommendations** for real estate companies: emphasize premium traits in marketing (size, age, terraced layouts), use model-backed explanations to improve transparency with clients, and combine ML predictions with expert appraisal for high-value listings.

---

## Key Question  
Which apartment features most influence sale prices in Daegu, and how accurately can machine learning models predict these prices to improve real estate pricing strategies?  

---

## Data
The analysis uses real-world housing data from Daegu, South Korea. This dataset captures essential apartment characteristics such as location features, building details, nearby facilities, and pricing information. It provides a comprehensive view of the factors that influence apartment sale prices in the city. You can access the dataset [here](https://drive.google.com/drive/folders/1fmkyfjrzuJNaH02sXhp5vUxqum9bH0Fx).

Key Features:
| Column                              | Description                                                                 |
| ----------------------------------- | --------------------------------------------------------------------------- |
| `HallwayType`                       | Type of apartment (e.g., hallway, penthouse, etc.).                         |
| `TimeToSubway`                      | Estimated walking time to the nearest subway station (e.g., 0–5 min, 10–15 min). |
| `SubwayStation`                     | Name of the nearest subway station.                                          |
| `N_FacilitiesNearBy(ETC)`           | Number of miscellaneous facilities near the apartment.                       |
| `N_FacilitiesNearBy(PublicOffice)`  | Number of nearby public office facilities.                                   |
| `N_SchoolNearBy(University)`        | Number of universities near the apartment.                                   |
| `N_Parkinglot(Basement)`            | Number of available basement parking lots.                                   |
| `YearBuilt`                         | Year when the apartment building was constructed.                            |
| `N_FacilitiesInApt`                 | Number of facilities available within the apartment itself.                  |
| `Size(sqft)`                        | Size of the apartment in square feet.                                        |
| `SalePrice`                         | Sale price of the apartment in Korean Won (target variable).                 |


---

## Data Cleaning
Before building predictive models, I carefully explored and prepared the dataset to ensure accuracy and consistency. This process includes:

- **Encoding categorical and ordinal variables**: Converted `HallwayType`, `TimeToSubway`, and `SubwayStation` into machine-readable formats using a mix of One-Hot and Ordinal Encoding.  
- **Scaling numerical features**: Applied different scalers to stabilise feature ranges (e.g., RobustScaler for skewed features like `Landsize` and `BuildingArea`, MinMaxScaler for `YearBuilt`).
- **Removing duplicates**: Eliminated 1,422 duplicate rows (~34%), leaving 2,701 valid records. This prevents the model from overemphasising repeated patterns and ensures fair representation of all properties.
- **Ensuring logical consistency**: Validated relationships across features (e.g., apartments with a listed basement parking lot must have at least one parking spot).  

After cleaning, I obtained a reliable dataset that maintained both data quality and representativeness, ready for exploratory analysis and modeling.

---

## Summary of Key Findings

### 1. Price Distribution
![Price Distribution](https://drive.google.com/uc?export=view&id=19MgCT3EKoSfX7V_JVQST0w4CnK8mg3u9)

- Apartment prices are **right-skewed**, meaning most properties are priced on the lower to mid-range side.  
- The majority of listings fall between **₩100k and ₩300k**.  
- A **noticeable peak around ₩200k** suggests this is the typical mid-range apartment price.  
- Higher-priced listings above **₩400k are less common**, with a long tail extending toward **₩600k**.

---

### 2. Hallway Type vs Price
![Hallway Type vs Price](https://drive.google.com/uc?export=view&id=1EVrFTs0xCj3mqafDVvbdw7YukD5LWANk)

- Properties with a **Terraced hallway type** have the **highest average sale prices**.  
- **Mixed hallway types** come next, followed by **Corridor types**.  
- This pattern suggests that **hallway design may reflect building quality or unit appeal**.  
- **Terraced layouts could be seen as more premium or desirable**, making them a potential value driver in pricing analysis.

---

### 3. Time To Subway vs Price
![Subway Time vs Price](https://drive.google.com/uc?export=view&id=1kCH9VtHh828NvCmtV4CZ_4B5RD0V6LtG)

- Properties within a **0–5 minute walk to a subway** have the **highest median prices** and the **most expensive outliers**.  
- **15–20 minute walk properties** generally sell for **less**.  
- Homes with **no nearby bus stop** also show **lower prices**.  
- Confirms the real estate insight that **proximity to transit significantly boosts property value**.  

---

### 4. Subway Station vs Price
![Subway vs Price](https://drive.google.com/uc?export=view&id=1dCkZzwe5wZpjnVwYIDK-YsTOWNsdohJd)

- **Banwoldang** and **Kyungbuk Uni Hospital** stations show the **highest average sale prices**, likely due to central location or access to key facilities.  
- **Chil-sung Market** and **Daegu** areas record **lower prices**, suggesting reduced demand or distance from city hotspots.  
- Properties with **no nearby subway** still show **relatively high prices**, possibly reflecting perks like parking availability or quieter neighbourhoods.  
- Confirms that **location near certain stations strongly influences property value**.

---

### 5. Number of Nearby Facilities vs Price
![Nearby Facilities vs Price](https://drive.google.com/uc?export=view&id=1PsNry2c1SNZHa-hDogoKT4XALUDyEL3T)

- Properties with **zero nearby facilities** have the **highest median prices** and a **wider price range**, suggesting they may be in premium residential zones where buyers value peace and exclusivity.  
- As the **number of facilities increases**, the **median price slightly decreases**, and the **price spread narrows**, indicating less variation in property values.  
- More nearby facilities **do not necessarily boost property value** as buyers may be prioritising **space and quietness over convenience**.

---

### 6. Number of Nearby Public Offices vs Price
![Offices vs Price](https://drive.google.com/uc?export=view&id=1go2bi1Vims3nxRuuIm_1Z3tPKkuFajRs)

- Properties with **0 or 4 nearby public offices** show some of the **highest median sale prices**, while those with more offices often trend lower.  
- The **price spread is widest** at 3, 4, and 7 offices, pointing to inconsistency in property values in those areas.  
- Having more public offices nearby **does not necessarily increase property value** as buyers may prioritise other factors like **neighbourhood quality or property size**.

---

### 7. Number of Nearby Universities vs Price
![Universities vs Price](https://drive.google.com/uc?export=view&id=1pJgeo4KEUUTnlHpt53i3XiDZCi3e6M8x)

- Properties with **1–2 nearby universities** show **higher and more consistent sale prices**, suggesting stronger demand in those areas.  
- Apartments with **4 or more universities nearby** tend to have **lower median prices** and a tighter range, possibly reflecting smaller or more standardised housing.  
- Properties with **zero nearby universities** have the **widest price spread**, including high-priced outliers, which may represent larger suburban homes less dependent on university proximity.

---

### 8. Number of Parking Spaces vs Price
![Parking vs Price](https://drive.google.com/uc?export=view&id=11GgLE7zQ5DANgr8L6NydVJCmZGZw4yE3)

- A **slight positive correlation** exists between the number of basement parking spaces and sale price, suggesting parking capacity adds some value.  
- **Most properties are tightly clustered**, indicating a typical range of prices regardless of parking variation.  
- **Outliers (highlighted using IQR method)** show unusually high sale prices, standing out from the general trend.  
- While the effect isn’t very strong, parking availability still appears to **play a role in apartment pricing**.  

---

### 9. Year Built vs Price
![Year vs Price](https://drive.google.com/uc?export=view&id=1bAlt-0xPfu1FGmpP6fc6P1YMkdjCdU8Z)

- A **general upward trend** shows that newer properties (built more recently) tend to have **higher sale prices**.  
- **High-price outliers**, especially around **2007**, spike above ₩600k, likely due to premium developments or special cases.  
- **Older homes (pre-1990)** sold at moderate prices but **did not reach the peaks** of newer properties.  
- Overall, **newer buildings are more valuable**, though outliers highlight that certain developments can **exceed typical market ranges**.  

---

### 10. Number of In-Apt Facilities vs Price
![Facilities vs Price](https://drive.google.com/uc?export=view&id=1GPFw4maCDODzxEDr5UqZdP2B1NCpOD61)

- A **clear upward trend** shows that more apartment facilities are generally linked to **higher sale prices**.  
- Apartments with **1–2 in-apt facilities** usually sell for **under ₩150k**.  
- **8–10 facilities** apartments' prices often reach **₩300k+**, with some exceeding **₩500k**.  
- The **price spread widens** as facilities increase, showing higher variability in well-equipped apartments.  
- **Outliers** are present across all categories but more common in **high-amenity units**, reflecting premium property spikes.

---

### 11. Size vs Price
![Size vs Price](https://drive.google.com/uc?export=view&id=1QZnvDWULzRCWgAAALvu9b2x8qAFJe0QK)

- A **clear positive relationship** exists between **apartment size and sale price** which means larger units generally sell for more.  
- The **trendline** supports this, confirming size as a strong driver of value.  
- **Outliers** show unusually high prices for their size, often pointing to **luxury units or special features**.  
- Most outliers are on the **high-price side**, suggesting certain properties exceed market norms due to unique factors.  

---

## Features Used
Based on the exploratory findings, the following features were selected for modeling, as they showed meaningful relationships with apartment sale prices:  

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

---

## Model Benchmarking  
Models tested include:  
- **Random Forest**  
- **Gradient Boosting**  
- **XGBoost**  
- **CatBoost**  
- **HistGradient Boosting**  
- **Ridge Regression**  
- **SVR**  

**XGBoost outperformed all others**, achieving on test data:  
| Metric                  | Score                |
|--------------------------|----------------------|
| **R²** (log-transformed) | 0.787                |
| **RMSE**                 | ₩47,530              |
| **MAE**                  | ₩38,060              |
| **MAPE**                 | 18.78%               |

Compared to a grouped mean baseline by size, XGBoost reduced MAE by over 55% and improved R² from -0.02 to 0.79.  

---

## Evaluation Metrics
I used these metrics to determine the best model:

- **R² (log-transformed):** Measures variance explained; good for overall model fit.  
- **RMSE:** Penalises large errors; important for high-priced apartments.  
- **MAE:** Average error in ₩; easy to interpret for stakeholders.  
- **MAPE:** Relative error in %; allows comparison across price ranges.  

These four metrics give a balanced view of model accuracy, capturing both statistical performance and business interpretability.

---

## Cross Validation

### Before Tuning
| Model            | Mean MAE (₩) | Mean RMSE (₩) | Mean R² (log) | Mean MAPE (%) |
|------------------|--------------|---------------|---------------|---------------|
| Random Forest    | 36,789.64    | 46,253.75     | 0.8289        | 18.31         |
| Gradient Boosting| 36,721.98    | 46,149.22     | 0.8302        | 18.29         |
| XGBoost          | 36,818.84    | 46,350.97     | 0.8285        | 18.30         |
| CatBoost         | 36,749.42    | 46,286.09     | 0.8292        | 18.27         |
| HistGBR          | 36,904.22    | 46,325.04     | 0.8277        | 18.41         |
| Ridge            | 38,299.26    | 47,853.93     | 0.8194        | 19.01         |
| SVR              | 36,563.55    | 46,482.78     | 0.8291        | 18.06         |


### After Tuning
| Model            | Mean MAE (₩) | Mean RMSE (₩) | Mean R² (log) | Mean MAPE (%) |
|------------------|--------------|---------------|---------------|---------------|
| Random Forest    | 36,818.70    | 46,269.41     | 0.8288        | 18.32         |
| Gradient Boosting| 36,622.36    | 46,109.94     | 0.8303        | 18.24         |
| XGBoost          | 36,652.90    | 46,080.20     | 0.8308        | 18.24         |
| CatBoost         | 36,693.64    | 46,192.38     | 0.8299        | 18.27         |
| HistGBR          | 36,801.40    | 46,297.99     | 0.8288        | 18.29         |
| Ridge            | 38,102.84    | 47,716.54     | 0.8199        | 18.93         |
| SVR              | 36,336.32    | 47,275.29     | 0.8257        | 17.81         |


### Results on Test Data
| Model            | Mean MAE (₩) | Mean RMSE (₩) | Mean R² (log) | Mean MAPE (%) |
|------------------|--------------|---------------|---------------|---------------|
| Random Forest    | 38,230.60    | 47,788.07     | 0.7850        | 18.88         |
| Gradient Boosting| 38,252.57    | 47,714.20     | 0.7857        | 18.90         |
| XGBoost          | 38,059.96    | 47,530.27     | 0.7873        | 18.78         |
| CatBoost         | 38,109.76    | 47,688.62     | 0.7859        | 18.98         |
| HistGBR          | 38,266.63    | 47,760.06     | 0.7853        | 18.87         |
| Ridge            | 39,635.57    | 48,863.41     | 0.7752        | 19.51         |
| SVR              | 38,026.95    | 48,768.05     | 0.7761        | 18.68         |

- Ensemble models consistently outperformed linear and SVR approaches  
- XGBoost captured complex feature interactions more effectively than Gradient Boosting and CatBoost

---

## Residual Analysis (Limitation)
![Residual](https://drive.google.com/uc?export=view&id=1BwkHvGI8U6QbcN-OWcMMMKj5l_YPUqnA)

- **Most predictions are accurate**  
  - Majority of points fall near the zero residual line  
  - Classified as "Accurate" (brown dots), within the acceptable MAPE threshold  

- **Underpricing trends**  
  - Positive residuals (points above zero line) indicate underestimation  
  - Concentrated in lower to mid-price brackets (~₩100k–₩300k)  
  - Suggests the model tends to undervalue properties in these ranges  

- **Overpricing trends**  
  - Negative residuals (points below zero line) indicate overestimation  
  - Spread across all price ranges but especially dense in the mid-segment (~₩200k–₩350k)  
  - Highlights model tendency to overpredict in middle market properties
 
---

## SHAP Values
![SHAP](https://drive.google.com/uc?export=view&id=1OWE248_e8siiTzGzK1ou_munwPbHztlc)

SHAP interpretability highlighted:  
- **Size(sqf)** as the most impactful driver of price  
- **YearBuilt** (newer = higher value) and **Terraced hallways** boosting desirability  
- Localised, sometimes counterintuitive effects from basement parking and facility counts  
- Nuanced patterns, where larger apartments weren’t always more expensive if combined with certain limiting features  

---

## Actionable Recommendations  

### 1. Pricing Oversight for High-End Properties
- Combine **model predictions with expert appraisals** for listings above ₩250,000  
- Use **residual analysis** to flag properties where prediction uncertainty is higher  
- Apply **tiered confidence levels** (standard vs. premium checks) based on price range  

### 2. Marketing & Sales Strategy
- Highlight **premium features** such as larger size, newer builds, and terraced hallways  
- Create **targeted marketing materials** that showcase these traits in listings  
- Use model insights to **rank and prioritise high-value apartments** for promotions  

### 3. Client Transparency & Trust
- Integrate **SHAP explainability plots** into sales presentations  
- Provide **clear, data-backed justifications** for price recommendations  
- Use explainability to **differentiate service quality** from competitors  

### 4. Data Enrichment & Model Refinement
- Incorporate **external factors** such as school quality, crime rates, and urban development plans  
- Collect more **granular location data** (neighbourhood-level features) for improved accuracy  
- Regularly **retrain models** to adapt to evolving market dynamics   

---

## Impact  
This project demonstrates how applied machine learning can directly support **business strategy and customer trust**:  
- **For real estate companies:** Improves accuracy of pricing, supports negotiations, and highlights features to promote in listings.  
- **For buyers and sellers:** Offers transparent, data-backed reasoning for price estimates, reducing uncertainty.  

By reducing pricing errors by more than **55% over a simple baseline**, this project illustrates not only technical skills in ML but also an understanding of how to transform predictions into **practical business value**.

---

- Presentation: [Daegu Apartments Presentation](https://drive.google.com/file/d/1TWmC_K0CRxsGQgrbf8VgNIQ5MCyMQruM/view?usp=sharing)
- Streamlit: [Daegu Apartments Price Predictor](https://module-03-capstone-risma-daegu-apartments.streamlit.app)
- Gradio: [Daegu Apartments Price Predictor](https://rismawidiya-portfolio-project.hf.space)
