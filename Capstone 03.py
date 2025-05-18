import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import joblib
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler

# Load model once globally:
model = pickle.load(open('gb_daegu_apartments_pipeline.sav', 'rb'))

# Load preprocessor once (fitted on training data offline):
preprocessor = joblib.load('preprocessor_daegu_apartments.pkl')

# Sidebar navigation for multipage:
page = st.sidebar.radio(
    "Hop on the Daegu Deals Express! Choose your stop:",
    ["Home Base", "Data Deep Dive", "Price Predictor", "Final Thoughts & Tips", "Explainable AI (SHAP)"]
)

def home():
    st.title("Welcome to Daegu Deals!")
    st.markdown("""
    ## Know Before You Sell
    
    In a fast-growing city like Daegu, pricing apartments right can be tricky.  
    Whether you're an individual seller or a real estate agent, setting the perfect price is key — too high and your property may linger on the market, too low and you lose out on profit.
    
    That's where this app comes in. Using machine learning, it predicts apartment prices by analysing factors like size, location, nearby amenities, and subway access. This helps you price smarter, sell faster, and maximise your returns.
    
    **What you can do here:**  
    - Upload your own dataset on the predictor page for quick batch predictions  
    - Explore detailed data analysis and visual insights  
    - Discover practical tips and recommendations to boost your apartment's value
    
    Ready to take the guesswork out of pricing? Let's dive in!
    """)

def data_analysis():
    st.title("Data Deep Dive: What's Really Driving Apartment Prices in Daegu?")
    st.markdown("Explore the hidden stories behind square footage, hallway types, subway stations, and more.")

    # Load data:
    data = pd.read_csv('data_daegu_apartment.csv')
    st.write("Quick Peek at the Dataset:", data.head())

    # Bar Chart | Hallway Type:
    st.subheader("Hallway Types vs Apartment Prices: Does the Layout Pay Off?")
    avg_price_by_hallway = data.groupby('HallwayType', as_index=False)['SalePrice'].mean()
    fig = px.bar(
        avg_price_by_hallway, x='HallwayType', y='SalePrice',
        labels={'SalePrice': 'Average Sale Price (₩)'},
        color='HallwayType',
        color_discrete_map={
            'Terraced': '#FF8DA1',
            'Mixed': '#F77896',
            'Corridor': '#FFB6C1'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # Box Plot | Time to Subway:
    st.subheader("Subway Distance vs Apartment Prices: How Far Is Too Far?")
    fig = px.box(data, x='TimeToSubway', y='SalePrice')
    fig.update_traces(marker_color='#FF8DA1')
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart | Subway Station:
    st.subheader("Stop vs Apartment Prices: Which Subway Stations Are Real Estate Hotspots?")
    avg_price_by_station = data.groupby('SubwayStation', as_index=False)['SalePrice'].mean()
    fig = px.bar(
        avg_price_by_station, x='SubwayStation', y='SalePrice',
        labels={'SalePrice': 'Average Sale Price (₩)'},
        color='SubwayStation'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Reusable Box Plot:
    def box_plot(x_col, title):
        fig = px.box(data, x=x_col, y='SalePrice', title=title)
        fig.update_traces(marker_color='#FF8DA1')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Convenience Counts: The Cost of Nearby Amenities")
    box_plot('N_FacilitiesNearBy(ETC)', "Nearby Facilities vs Apartment Prices: More Shops, More Won?")
    box_plot('N_FacilitiesNearBy(PublicOffice)', "Nearby Public Offices vs Apartment Prices: Do Public Offices Boost Prices?")
    box_plot('N_SchoolNearBy(University)', "Nearby Universities vs Apartment Prices: Are Apartments Near Universities Worth More?")

    st.subheader("Inside Scoop: Do More In-Apt Facilities Mean Higher Prices?")
    box_plot('N_FacilitiesInApt', "Apartment Facilities vs Apartment Prices: Do More Facilities Mean Higher Prices?")

    # Scatter Plot with Trendline and Outliers:
    def scatter_outlier_plot(x_col, title):
        Q1 = data['SalePrice'].quantile(0.25)
        Q3 = data['SalePrice'].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data['SalePrice'] < lower) | (data['SalePrice'] > upper)]

        fig = px.scatter(
            data, x=x_col, y='SalePrice',
            title=title,
            hover_data=['HallwayType', 'TimeToSubway', 'SubwayStation', 'YearBuilt'],
            trendline='ols',
            trendline_color_override='red'
        )
        fig.update_traces(marker=dict(color='#FF8DA1', size=8))

        fig.add_scatter(
            x=outliers[x_col], y=outliers['SalePrice'], mode='markers',
            marker=dict(color='#FF6F91', size=10, symbol='x'), name='Outliers'
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Parking, Age, Size: What Really Moves the Price Tag?")
    scatter_outlier_plot('N_Parkinglot(Basement)', "Basement Parking Spaces vs Apartment Prices: Do More Basement Parking Spaces Drive Up Prices?")
    scatter_outlier_plot('YearBuilt', "Year Built vs Apartment Prices: How Much Does Year Built Matter?")
    scatter_outlier_plot('Size(sqf)', "Apartment Size vs Apartment Prices: Bigger Means Pricier? Let's See!")

    # Parallel Coordinates:
    st.subheader("Multi-Factor Analysis: What Factors Cluster with Price?")
    fig = px.parallel_coordinates(
        data,
        dimensions=[
            'Size(sqf)', 'YearBuilt', 'N_FacilitiesInApt',
            'N_FacilitiesNearBy(ETC)', 'N_SchoolNearBy(University)'
        ],
        color='SalePrice',
        color_continuous_scale=['#FFE5EC', '#FFB3D1', '#FF5CA8', '#C9184A', '#86002D'],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Faceted Plot:
    st.subheader("Hallway Showdown: Does Layout Affect Price Per Size?")
    fig = px.scatter(
        data, x='Size(sqf)', y='SalePrice', color='HallwayType',
        facet_col='HallwayType',
        color_discrete_sequence=['#FF8DA1', '#F77896', '#FFB6C1'],
        trendline='ols'
    )
    st.plotly_chart(fig, use_container_width=True)

def explainable_ai():
    st.title("Explainable AI: SHAP Insights")
    st.markdown("""
    Curious how the model makes its predictions?  
    This page breaks it down using **SHAP (SHapley Additive exPlanations)** — a method to explain how much each feature contributes to the final predicted price.
    """)

    X_test_selected = pd.read_csv("X_test_selected.csv")

    # Use already loaded model:

    explainer = shap.Explainer(model, X_test_selected)
    shap_values = explainer(X_test_selected)

    st.subheader("Top Features Impacting Predictions")
    plt.figure()
    shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(plt.gcf())

    st.subheader("Feature Contributions Across Samples")
    plt.figure()
    shap.summary_plot(shap_values, X_test_selected, show=False)
    st.pyplot(plt.gcf())

    st.subheader("Detailed SHAP Waterfall (Sample 25 and 35)")
    for idx in [25, 35]:
        st.markdown(f"### SHAP Waterfall Plot – Index {idx}")
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        st.pyplot(plt.gcf())

    st.markdown("---")
    st.info("Note: These visualisations explain the **trained model's logic** on held-out test data. They're not generated in real-time for uploaded inputs.")

def price_predictor():
    st.title("Daegu Apartment Price Predictor")
    st.markdown("Input apartment details manually or upload your own dataset for batch predictions.")

    # Adjusted input options to match ordinal categories:
    hallway = st.sidebar.selectbox("What's the Hallway Vibe?", ['Corridor', 'Mixed', 'Terraced'])
    subway = st.sidebar.selectbox("Closest Subway Stop?", ['No Subway Nearby', 'Bangoge', 'Banwoldang', 'Chil-sung Market', 'Daegu', 'Kyungbuk Uni Hospital', 'Myung-duk', 'Sin-nam'])
    time_to_subway = st.sidebar.selectbox("Time to the Tracks?", ['No Bus Stop Nearby', '10-15min', '5-10min', '0-5min'])  # same order as ordinal encoder categories

    col1, col2 = st.columns(2)
    with col1:
        fac_etc = st.slider("Miscellaneous Nearby Perks (ETC)", 0, 5, value=2)
        fac_uni = st.slider("University Count Nearby", 0, 5, value=2)
        year_built = st.slider("Year the Apartment Was Built", 1978, 2015, value=2006)
        fac_in_apt = st.slider("Cool Facilities Inside the Apartment", 1, 10, value=5)
    with col2:
        fac_office = st.slider("Nearby Public Offices", 0, 7, value=4)
        parking_bsmnt = st.slider("Basement Parking Spaces", 0, 1400, value=536, step=10)
        size = st.slider("Apartment Size (sqft)", 135, 2337, value=910)

    if st.button("Crunch the Numbers!"):
        input_data = pd.DataFrame({
            'HallwayType': [hallway],
            'TimeToSubway': [time_to_subway],
            'SubwayStation': [subway],
            'N_FacilitiesNearBy(ETC)': [fac_etc],
            'N_FacilitiesNearBy(PublicOffice)': [fac_office],
            'N_SchoolNearBy(University)': [fac_uni],
            'N_Parkinglot(Basement)': [parking_bsmnt],
            'YearBuilt': [year_built],
            'N_FacilitiesInApt': [fac_in_apt],
            'Size(sqf)': [size]
        })

        # Use transform only, do NOT fit again:
        try:
            X_encoded = preprocessor.transform(input_data)
            prediction = model.predict(X_encoded)[0]
            prediction_m = prediction / 1_000_000

            st.markdown(f"""
            ### **Your Apartment Might Sell For:**  
            ## <span style='color:#d63384'>₩{prediction_m:,.2f}M</span>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Batch Prediction: Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Upload CSV with apartment data", type="csv")

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", batch_df.head())

        try:
            # Use transform only here as well:
            X_batch_encoded = preprocessor.transform(batch_df)
            batch_pred = model.predict(X_batch_encoded)
            batch_df['Predicted_Price'] = batch_pred
            st.write("Here are the predicted prices for your apartments:")
            st.dataframe(batch_df)

            csv = batch_df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv, "daegu_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")

def conclusion():
    st.title("Final Thoughts & Tips for Daegu Apartment Sellers")
    st.markdown("""
    ### Insights from the Data and Model:
    - **HallwayType Matters:** Terraced hallways generally fetch higher prices.
    - **Proximity to Subway:** Apartments closer to subway stations sell better and for more.
    - **Facilities Inside and Nearby:** More amenities both inside and nearby increase value.
    - **Age & Size:** Newer and bigger apartments command higher prices, as expected.
    
    ### Recommendations:
    - **Stage Your Apartment:** Invest in in-apartment facilities and upkeep to boost appeal.
    - **Highlight Convenience:** Emphasize closeness to transit and nearby amenities in listings.
    - **Price Smartly:** Use the model to set competitive yet profitable prices.
    - **Focus on Layout:** Terraced hallway apartments might deserve premium pricing.
    
    ### Thanks for exploring Daegu Deals!  
    Ready to make smart selling decisions? Try the Price Predictor page or upload your own data for instant insights.
    """)

# Page router:
if page == "Home Base":
    home()
elif page == "Data Deep Dive":
    data_analysis()
elif page == "Price Predictor":
    price_predictor()
elif page == "Final Thoughts & Tips":
    conclusion()
elif page == "Explainable AI (SHAP)":
    explainable_ai()