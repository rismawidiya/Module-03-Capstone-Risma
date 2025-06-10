import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import joblib
from streamlit_option_menu import option_menu

# Load pipeline:
model_pipeline = pickle.load(open('xgb_daegu_apartments_pipeline.sav', 'rb'))

# Sidebar navigation for multipage:
with st.sidebar:
    selected = option_menu(
        "Hop on the Daegu Deals Express!",
        ["Home Base", "Data Deep Dive", "Price Predictor", "Disclaimer"],
        icons=["house", "search", "cash-stack", "exclamation-triangle"],
        menu_icon="building",
        default_index=0,
        orientation="vertical"
    )

    # Space:
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Creator info at the bottom:
    st.markdown(
        """
        <div style='
            margin-top: 25vh;
            text-align: center;
            color: #FFFFFF;
            font-size: 14px;
            font-weight: italic;
        '>
            Created by Risma W. P.
        <div style='margin-top:8px;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-flower3" viewBox="0 0 16 16">
            <path d="M11.424 8c.437-.052.811-.136 1.04-.268a2 2 0 0 0-2-3.464c-.229.132-.489.414-.752.767C9.886 4.63 10 4.264 10 4a2 2 0 1 0-4 0c0 .264.114.63.288 1.035-.263-.353-.523-.635-.752-.767a2 2 0 0 0-2 3.464c.229.132.603.216 1.04.268-.437.052-.811.136-1.04.268a2 2 0 1 0 2 3.464c.229-.132.489-.414.752-.767C6.114 11.37 6 11.736 6 12a2 2 0 1 0 4 0c0-.264-.114-.63-.288-1.035.263.353.523.635.752.767a2 2 0 1 0 2-3.464c-.229-.132-.603-.216-1.04-.268M9 4a2 2 0 0 1-.045.205q-.059.2-.183.484a13 13 0 0 1-.637 1.223L8 6.142l-.135-.23a13 13 0 0 1-.637-1.223 4 4 0 0 1-.183-.484A2 2 0 0 1 7 4a1 1 0 1 1 2 0M3.67 5.5a1 1 0 0 1 1.366-.366 2 2 0 0 1 .156.142q.142.15.326.4c.245.333.502.747.742 1.163l.13.232-.265.002a13 13 0 0 1-1.379-.06 4 4 0 0 1-.51-.083 2 2 0 0 1-.2-.064A1 1 0 0 1 3.67 5.5m1.366 5.366a1 1 0 0 1-1-1.732l.047-.02q.055-.02.153-.044.202-.048.51-.083a13 13 0 0 1 1.379-.06q.135 0 .266.002l-.131.232c-.24.416-.497.83-.742 1.163a4 4 0 0 1-.327.4 2 2 0 0 1-.155.142M9 12a1 1 0 0 1-2 0 2 2 0 0 1 .045-.206q.058-.198.183-.483c.166-.378.396-.808.637-1.223L8 9.858l.135.23c.241.415.47.845.637 1.223q.124.285.183.484A1.3 1.3 0 0 1 9 12m3.33-6.5a1 1 0 0 1-.366 1.366 2 2 0 0 1-.2.064q-.202.048-.51.083c-.412.045-.898.061-1.379.06q-.135 0-.266-.002l.131-.232c.24-.416.497-.83.742-1.163a4 4 0 0 1 .327-.4q.07-.074.114-.11l.041-.032a1 1 0 0 1 1.366.366m-1.366 5.366a2 2 0 0 1-.155-.141 4 4 0 0 1-.327-.4A13 13 0 0 1 9.74 9.16l-.13-.232.265-.002c.48-.001.967.015 1.379.06q.308.035.51.083.098.024.153.044l.048.02a1 1 0 1 1-1 1.732zM8 9a1 1 0 1 1 0-2 1 1 0 0 1 0 2"/>
            </svg>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

def home():
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:10px;
            border-radius:10px;
            font-size:40px;
            font-weight:bold;
            text-align:center;
        '>
            Welcome to Daegu Deals!
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    > **Daegu is growing fast**, and apartments are one of the most in-demand types of housing. But figuring out how much a unit should sell for isn't always easy. A price that's too high could scare off buyers. Too low, and real estate company might leave money on the table.  
    >  
    > That's where this app comes in. Instead of relying purely on gut feeling or rough comparisons, we can use machine learning to analyse key property features—like size, age, location, and nearby amenities—to predict prices more accurately. This gives real estate company a solid, data-backed starting point when listing a property.  
    >  
    > **What you can do here:**  
    > • Explore detailed data analysis and visual insights  
    > • Upload your own dataset on the predictor page for quick batch predictions  
    >  
    > **Ready to take the guesswork out of pricing? Let's dive in!**
    """)

def data_analysis():
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            What's Really Driving Apartment Prices in Daegu?
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    > Explore the hidden stories behind square footage, hallway types, subway stations, and more.
    """)

    # Load data:
    data = pd.read_csv('data_daegu_apartment.csv')
    st.write("Quick Peek at the Dataset:", data.head())

    # Ensure HallwayType values are clean:
    data['HallwayType'] = data['HallwayType'].str.strip().str.title()

    # Bar Chart | Hallway Type:
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Hallway Types vs Apartment Prices:<br>
            Does the Layout Pay Off?
        </div>
        """,
        unsafe_allow_html=True
    )
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

    # Manually map TimeToSubway values using your predefined mapping:
    time_rename_map = {
        '5min~10min': '5min-10min',
        '10min~15min': '10min-15min',
        '15min~20min': '15min-20min',
        'no_bus_stop_nearby': 'No Bus Stop Nearby'
    }

    # Apply the mapping to the TimeToSubway column:
    data['TimeToSubway'] = data['TimeToSubway'].replace(time_rename_map)

    # Order the TimeToSubway values:
    time_order = ['0-5min', '5min-10min', '10min-15min', '15min-20min', 'No Bus Stop Nearby']
    
    # Box Plot | Time to Subway:
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Subway Distance vs Apartment Prices:<br>
            How Far Is Too Far?
        </div>
        """,
        unsafe_allow_html=True
    )
    fig = px.box(
        data, x='TimeToSubway', y='SalePrice',
        category_orders={'TimeToSubway': time_order}
    )
    fig.update_traces(marker_color='#FF8DA1')
    st.plotly_chart(fig, use_container_width=True)

    # Manually map SubwayStation values using your predefined mapping:
    subway_rename_map = {
        'Myung-duk': 'Myung-duk',
        'Kyungbuk_uni_hospital': 'Kyungbuk Uni Hospital',
        'Sin-nam': 'Sin-nam',
        'Banwoldang': 'Banwoldang',
        'Bangoge': 'Bangoge',
        'no_subway_nearby': 'No Subway Nearby',
        'Chil-sung-market': 'Chil-sung Market',
        'Daegu': 'Daegu'
    }

    # Apply the mapping to the SubwayStation column:
    data['SubwayStation'] = data['SubwayStation'].map(subway_rename_map)

    # Custom pink colour palette:
    custom_pinks = ['#FFC0CB', '#FFB6C1', '#FF69B4', '#FF1493', '#DB7093', '#C71585', '#E75480', '#F8BBD0']

    # Bar Chart | Subway Station:
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Stop vs Apartment Prices:<br>
            Which Subway Stations Are Real Estate Hotspots?
        </div>
        """,
        unsafe_allow_html=True
    )
    avg_price_by_station = data.groupby('SubwayStation', as_index=False)['SalePrice'].mean()
    fig = px.bar(
        avg_price_by_station, x='SubwayStation', y='SalePrice',
        labels={'SalePrice': 'Average Sale Price (₩)'},
        color='SubwayStation',
        color_discrete_sequence=custom_pinks

    )
    st.plotly_chart(fig, use_container_width=True)

    # Reusable Box Plot:
    def box_plot(x_col, title):
        fig = px.box(data, x=x_col, y='SalePrice', title=title)
        fig.update_traces(marker_color='#FF8DA1')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Convenience Counts:<br>
            The Cost of Nearby Amenities
        </div>
        """,
        unsafe_allow_html=True
    )
    box_plot('N_FacilitiesNearBy(ETC)', "Nearby Facilities vs Apartment Prices: More Shops, More Won?")
    box_plot('N_FacilitiesNearBy(PublicOffice)', "Nearby Public Offices vs Apartment Prices: Do Public Offices Boost Prices?")
    box_plot('N_SchoolNearBy(University)', "Nearby Universities vs Apartment Prices: Are Apartments Near Universities Worth More?")

    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Inside Scoop:<br>
            Do More In-Apt Facilities Mean Higher Prices?
        </div>
        """,
        unsafe_allow_html=True
    )
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

    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Parking, Age, Size:<br>
            What Really Moves the Price Tag?
        </div>
        """,
        unsafe_allow_html=True
    )
    scatter_outlier_plot('N_Parkinglot(Basement)', "Basement Parking Spaces vs Apartment Prices: Do More Basement Parking Spaces Drive Up Prices?")
    scatter_outlier_plot('YearBuilt', "Year Built vs Apartment Prices: How Much Does Year Built Matter?")
    scatter_outlier_plot('Size(sqf)', "Apartment Size vs Apartment Prices: Bigger Means Pricier? Let's See!")

    # Parallel Coordinates:
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Multi-Factor Analysis:<br>
            What Factors Cluster with Price?
        </div>
        """,
        unsafe_allow_html=True
    )
    fig = px.parallel_coordinates(
    data,
    dimensions=[
        'Size(sqf)', 'YearBuilt', 'N_FacilitiesInApt',
        'N_FacilitiesNearBy(ETC)', 'N_SchoolNearBy(University)'
    ],
    color='SalePrice',
    color_continuous_scale=['#FFE5EC', '#FFB3D1', '#FF5CA8', '#C9184A', '#86002D'],
)

    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        width=900
    )

    st.plotly_chart(fig, use_container_width=True)

    # Faceted Plot:
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:20px;
            border-radius:10px;
            font-size:30px;
            font-weight:bold;
            text-align:center;
        '>
            Space, Style, and Subway:<br>
            How Layout and Transit Access Shape Prices
        </div>
        """,
        unsafe_allow_html=True
    )
    custom_pink_scale = ["#CF3F59", '#F77896', "#5D1B25"]

    # Order the TimeToSubway values:
    time_order = ['0-5min', '5min-10min', '10min-15min', '15min-20min', 'No Bus Stop Nearby']

    # Create the figure:
    fig = px.scatter(
        data,
        x='Size(sqf)',
        y='SalePrice',
        color='HallwayType',
        facet_col='TimeToSubway',
        facet_col_wrap=3,
        color_discrete_sequence=custom_pink_scale,
        category_orders={'TimeToSubway': time_order},
        title='Sale Price vs Size Faceted by Subway Time and Hallway Type'
    )

    # Display in Streamlit:
    st.plotly_chart(fig, use_container_width=True)


def price_predictor():
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:10px;
            border-radius:10px;
            font-size:40px;
            font-weight:bold;
            text-align:center;
        '>
            Daegu Apartment Price Predictor
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    > Input apartment details manually or upload your own dataset for batch predictions.
    """)

    # Location Info:
    st.markdown("### Location Info")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        subway = st.selectbox("Closest Subway Stop?", [
            'No Subway Nearby', 'Bangoge', 'Banwoldang', 'Chil-sung Market',
            'Daegu', 'Kyungbuk Uni Hospital', 'Myung-duk', 'Sin-nam'
        ])
        time_to_subway = st.selectbox("Time to Subway?", ['No Bus Stop Nearby', '0-5min', '5min-10min', '10min-15min', '15min-20min'])
    with loc_col2:
        fac_office = st.slider("Nearby Public Offices", 0, 7, value=4)
        fac_etc = st.slider("Miscellaneous Nearby Facilities (ETC)", 0, 5, value=2)
        fac_uni = st.slider("Nearby Universities", 0, 5, value=2)

    # Apartment Features:
    st.markdown("### Apartment Features")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        hallway = st.selectbox("Hallway Type", ['Corridor', 'Mixed', 'Terraced'])
        year_built = st.slider("Year Built", 1978, 2015, value=2006)
        fac_in_apt = st.slider("In-Apartment Facilities", 1, 10, value=5)
    with feat_col2:
        parking_bsmnt = st.slider("Basement Parking Spaces", 0, 1321, value=536)
        size = st.slider("Apartment Size (sqft)", 135, 2337, value=910)

    # Prepare input dataframe:
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

    if st.button("Predict Sale Price"):
        try:
            # Use combined pipeline:
            prediction_log = model_pipeline.predict(input_data)[0]
            prediction = np.expm1(prediction_log)

            st.markdown(f"""
            ### **Estimated Sale Price:**  
            ## <span style='color:#d63384'>₩{prediction:,.0f}</span>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Batch prediction:
    st.markdown("---")
    st.markdown("### Batch Predictions")
    st.markdown("""
    > Upload your own file to get predictions for multiple apartments.
    """)

    uploaded_file = st.file_uploader("Upload CSV File:", type=['csv'])

    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)

            # Apply renaming to match the training data:
            if 'HallwayType' in user_data.columns:
                user_data['HallwayType'] = user_data['HallwayType'].str.strip().str.title()

            time_rename_map = {
                '5min~10min': '5min-10min',
                '10min~15min': '10min-15min',
                '15min~20min': '15min-20min',
                'no_bus_stop_nearby': 'No Bus Stop Nearby'
            }
            subway_rename_map = {
                'Myung-duk': 'Myung-duk',
                'Kyungbuk_uni_hospital': 'Kyungbuk Uni Hospital',
                'Sin-nam': 'Sin-nam',
                'Banwoldang': 'Banwoldang',
                'Bangoge': 'Bangoge',
                'no_subway_nearby': 'No Subway Nearby',
                'Chil-sung-market': 'Chil-sung Market',
                'Daegu': 'Daegu'
            }

            if 'TimeToSubway' in user_data.columns:
                user_data['TimeToSubway'] = user_data['TimeToSubway'].replace(time_rename_map)
            if 'SubwayStation' in user_data.columns:
                user_data['SubwayStation'] = user_data['SubwayStation'].replace(subway_rename_map)

            # Predict using the trained pipeline:
            predictions_log = model_pipeline.predict(user_data)
            predictions = np.expm1(predictions_log)

            # Add predictions to DataFrame:
            user_data['Predicted Sale Price (₩)'] = predictions
            user_data['Predicted Price (Million ₩)'] = predictions / 1_000_000

            st.markdown("### Predictions:")
            st.dataframe(user_data)

            # Download results:
            csv = user_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                data=csv,
                file_name="daegu_apartments_predictions.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Something went wrong while processing the file: {e}")


def conclusion():
    st.markdown(
        """
        <div style='
            background-color:#E79898;
            color:#FFFFFF;
            padding:10px;
            border-radius:10px;
            font-size:40px;
            font-weight:bold;
            text-align:center;
        '>
            Disclaimer & Important Notes
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    > - **Tricky High-End Pricing**: The model's accuracy tends to drop for luxury apartments priced above ₩250,000. Expert appraisal is strongly recommended for these properties.
    > - **Some Underpricing in Mid-Range**: A few apartments in the ₩100,000-₩300,000 range may be underestimated by the model.
    > - **Complex Feature Interactions**: Some feature effects are non-linear or unintuitive and should be carefully interpreted when making decisions.
    
    > **Please treat the predicted prices as guidance only. Final pricing decisions should always involve thorough human appraisal and market expertise.**

    > **Thanks for exploring Daegu Deals!**
    """)

# Page router:
if selected == "Home Base":
    home()
elif selected == "Data Deep Dive":
    data_analysis()
elif selected == "Price Predictor":
    price_predictor()
elif selected == "Disclaimer":
    conclusion()