import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the pipeline:
model = joblib.load("xgb_daegu_apartments_pipeline.sav")

def data_analysis():
    # Load data:
    data = pd.read_csv('data_daegu_apartment.csv')

    # Clean columns:
    data['HallwayType'] = data['HallwayType'].str.strip().str.title()

    # Mapping categorical columns:
    time_rename_map = {
        '5min~10min': '5min-10min',
        '10min~15min': '10min-15min',
        '15min~20min': '15min-20min',
        'no_bus_stop_nearby': 'No Bus Stop Nearby'
    }
    data['TimeToSubway'] = data['TimeToSubway'].replace(time_rename_map)

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
    data['SubwayStation'] = data['SubwayStation'].map(subway_rename_map)

    # Bar chart: Hallway Type vs SalePrice:
    hallway_avg = data.groupby('HallwayType')['SalePrice'].mean().reset_index()
    fig_hallway = px.bar(
        hallway_avg, x='HallwayType', y='SalePrice',
        color='HallwayType',
        color_discrete_map={
            'Terraced': '#FF8DA1',
            'Mixed': '#F77896',
            'Corridor': '#FFB6C1'
        },
        labels={'SalePrice': 'Average Sale Price (₩)'}
    )
    
    # Remove the internal title completely:
    fig_hallway.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Hallway Type vs Sale Price</h3>")

    # Display:
    gr.Plot(fig_hallway)

    # Box plot: Time to Subway:
    time_order = ['0-5min', '5min-10min', '10min-15min', '15min-20min', 'No Bus Stop Nearby']
    fig_time = px.box(
        data, x='TimeToSubway', y='SalePrice',
        category_orders={'TimeToSubway': time_order}
    )
    fig_time.update_traces(marker_color='#FF8DA1')

    # Remove the internal title completely:
    fig_time.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Subway Distance vs Apartment Price</h3>")

    # Display:
    gr.Plot(fig_time)

    # Bar chart: Subway Station:
    avg_price_by_station = data.groupby('SubwayStation')['SalePrice'].mean().reset_index()
    fig_subway = px.bar(
        avg_price_by_station, x='SubwayStation', y='SalePrice',
        color='SubwayStation',
        color_discrete_sequence=['#FFC0CB', '#FFB6C1', '#FF69B4', '#FF1493', '#DB7093', '#C71585', '#E75480', '#F8BBD0']
    )

    # Remove the internal title completely:
    fig_subway.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Subway Station vs Sale Price</h3>")

    # Display:
    gr.Plot(fig_subway)

    # Box plot: ETC:
    fig_etc = px.box(data, x="N_FacilitiesNearBy(ETC)", y="SalePrice")
    fig_etc.update_traces(marker_color="#FF8DA1")

    # Remove the internal title completely:
    fig_etc.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Nearby Facilities vs Apartment Price</h3>")

    # Display:
    gr.Plot(fig_etc)

    # Box plot: Offices:
    fig_office = px.box(data, x="N_FacilitiesNearBy(PublicOffice)", y="SalePrice")
    fig_office.update_traces(marker_color="#FF8DA1")

    # Remove the internal title completely:
    fig_office.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Nearby Public Offices vs Apartment Price</h3>")

    # Display:
    gr.Plot(fig_office)

    # Scatter Plot with outliers:
    Q1 = data['SalePrice'].quantile(0.25)
    Q3 = data['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data['SalePrice'] < lower) | (data['SalePrice'] > upper)]

    fig_size = px.scatter(data, x="Size(sqf)", y="SalePrice",
                      hover_data=["YearBuilt", "HallwayType"])
    fig_size.update_traces(marker=dict(color='#FF8DA1', size=8))
    fig_size.add_scatter(
        x=outliers["Size(sqf)"],
        y=outliers["SalePrice"],
        mode="markers",
        marker=dict(color="#FF6F91", size=10, symbol="x"),
        name="Outliers"
    )

    # Remove the internal title completely:
    fig_size.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Apartment Size vs Price</h3>")

    # Display:
    gr.Plot(fig_size)

    # Box plot: Facilities:
    fig_facilities = px.box(data, x="N_FacilitiesInApt", y="SalePrice")
    fig_facilities.update_traces(marker_color="#FF8DA1")

    # Remove the internal title completely:
    fig_facilities.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>In-Apt Facilities vs Price</h3>")

    # Display:
    gr.Plot(fig_facilities)

    # Parallel coordinates plot:
    fig_parallel = px.parallel_coordinates(
        data,
        dimensions=[
            'Size(sqf)', 'YearBuilt', 'N_FacilitiesInApt',
            'N_FacilitiesNearBy(ETC)', 'N_SchoolNearBy(University)'
        ],
        color='SalePrice',
        color_continuous_scale=['#FFE5EC', '#FFB3D1', '#FF5CA8', '#C9184A', '#86002D']
    )

    # Remove the internal title completely:
    fig_parallel.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Multivariate Influence on Apartment Price</h3>")

    # Display:
    gr.Plot(fig_parallel)

    # Custom colour palette:
    custom_pink_scale = ["#CF3F59", '#F77896', "#5D1B25"]

    # Order TimeToSubway values:
    time_order = ['0-5min', '5min-10min', '10min-15min', '15min-20min', 'No Bus Stop Nearby']

    # Create the faceted figure:
    fig_faceted = px.scatter(
        data,
        x='Size(sqf)',
        y='SalePrice',
        color='HallwayType',
        facet_col='TimeToSubway',
        facet_col_wrap=3,
        color_discrete_sequence=custom_pink_scale,
        category_orders={'TimeToSubway': time_order}
    )

    # Remove the internal title completely:
    fig_faceted.update_layout(title=None)

    # Add a centered, black Gradio title above the chart:
    gr.Markdown("<h3 style='text-align: center; color: black;'>Sale Price vs Size Faceted by Subway Time and Hallway Type</h3>")

    # Display:
    gr.Plot(fig_faceted)

    return "Explore the hidden stories behind square footage, hallway types, subway stations, and more."
    
# Define prediction function:
def predict_price(hallway, subway_time, station, size, year, facilities, univ_nearby, basement_parking, public_office, etc_facilities):
    try:
        input_dict = {
        "HallwayType": hallway,
        "TimeToSubway": subway_time,
        "SubwayStation": station,
        "Size(sqf)": size,
        "YearBuilt": year,
        "N_FacilitiesInApt": facilities,
        "N_SchoolNearBy(University)": univ_nearby,
        "N_Parkinglot(Basement)": basement_parking,
        "N_FacilitiesNearBy(PublicOffice)": public_office,
        "N_FacilitiesNearBy(ETC)": etc_facilities
    }

        df = pd.DataFrame([input_dict])
        pred_log = model.predict(df)[0]
        prediction = np.expm1(pred_log)
        return f"₩ {prediction:,.0f}"
    
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Dropdown values:
hallway_options = ["Corridor", "Mixed", "Terraced"]
subway_options = ["0-5min", "5min-10min", "10min-15min", "15min-20min", "No Bus Stop Nearby"]
station_options = ['No Subway Nearby', 'Bangoge', 'Banwoldang', 'Chil-sung Market', 'Daegu', 
                   'Kyungbuk Uni Hospital', 'Myung-duk', 'Sin-nam']

# Create Gradio interface:
with gr.Blocks(css="""
    body {
        background-color: #fff0f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .section-header {
        font-size: 20px; 
        font-weight: 700; 
        margin-bottom: 12px; 
        color: #e75480;
        border-bottom: 2px solid #f8bbd0;
        padding-bottom: 4px;
    }
    /* Range sliders */
    input[type=range]::-webkit-slider-thumb {
        background: #e75480;
    }
    input[type=range]::-moz-range-thumb {
        background: #e75480;
    }
    input[type=range]::-ms-thumb {
        background: #e75480;
    }
    input[type=range]::-webkit-slider-runnable-track {
        background: #f8bbd0;
    }
    input[type=range]::-moz-range-track {
        background: #f8bbd0;
    }
    input[type=range]::-ms-track {
        background: #f8bbd0;
        border-color: transparent;
        color: transparent;
    }
    #predict_btn {
        background-color: #e75480;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        margin-top: 15px;
        box-shadow: 0 4px 8px rgba(231, 84, 128, 0.3);
        transition: background-color 0.3s ease;
    }
    #predict_btn:hover {
        background-color: #c7446e;
        box-shadow: 0 6px 12px rgba(199, 68, 110, 0.5);
    }
    #prediction_output {
        background-color: #fce4ec;
        border: 2px solid #f8bbd0;
        border-radius: 12px;
        padding: 15px;
        font-size: 20px;
        font-weight: 700;
        color: #b71c1c;
        margin-top: 20px;
        text-align: center;
    }
    .gr-dropdown, .gr-slider, .gr-textbox {
        font-weight: 600;
        color: #880e4f;
    }
    /* Style all tab buttons */
    .tab-nav button {
        all: unset; /* Remove Gradio's inline styles */
        display: inline-block;
        font-weight: 700;
        font-size: 16px;
        padding: 10px 16px;
        cursor: pointer;
        background-color: transparent;
        color: #ff8da1;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
    }
    /* Hover state */
    .tab-nav button:hover {
        background-color: #fce4ec;
        color: #d0628f;
    }
    /* Selected/active tab */
    .tab-nav button[aria-selected="true"] {
        color: #c870a0;
        border-bottom: 3px solid #ff8da1;
        background-color: #fce4ec;
    }
""") as demo:
    gr.Markdown("""<h1 style='color: #ff8da1; text-align: center; font-size: 36px; margin-top: 20px; font-weight: bold;'>Daegu Deals: Predicting Apartment Prices in Daegu, South Korea</h1>
""")
    with gr.Tabs():
        with gr.Tab("Predict Price"):
            gr.Markdown('<div class="section-header">Location & Layout</div>')
            with gr.Row():
                hallway = gr.Dropdown(hallway_options, label="Hallway Type")
                subway_time = gr.Dropdown(subway_options, label="Time to Subway")
                station = gr.Dropdown(station_options, label="Nearest Subway Station")

            gr.Markdown('<div class="section-header">Size & Year Built</div>')
            with gr.Row():
                size = gr.Slider(135, 2337, step=1, label="Size (sqft)")
                year = gr.Slider(1978, 2015, step=1, label="Year Built")

            gr.Markdown('<div class="section-header">Amenities & Facilities</div>')
            with gr.Row():
                facilities = gr.Slider(0, 10, step=1, label="Facilities in Apartment")
                univ = gr.Slider(0, 5, step=1, label="Universities Nearby")
                parking = gr.Slider(0, 1321, step=1, label="Basement Parking Lots")
            with gr.Row():
                public_office = gr.Slider(0, 10, step=1, label="Public Offices Nearby")
                etc_facilities = gr.Slider(0, 10, step=1, label="ETC Facilities Nearby")

            predict_btn = gr.Button("Predict Price", elem_id="predict_btn")
            prediction_output = gr.Textbox(label="Estimated Sale Price", elem_id="prediction_output", interactive=False)

            predict_btn.click(
                fn=predict_price,
                inputs=[hallway, subway_time, station, size, year, facilities, univ, parking, public_office, etc_facilities],
                outputs=prediction_output
            )

        with gr.Tab("Explore Data"):
            description = data_analysis()
            gr.Markdown(description)

if __name__ == "__main__":
    demo.launch()
