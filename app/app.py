import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path

st.set_page_config(
    page_title="Home Energy Intelligence AI",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "energy_model.pkl"
model = joblib.load(MODEL_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def search_address(query: str):
    if not query or len(query.strip()) < 3:
        return []

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 5,
        "countrycodes": "us"
    }
    headers = {
        "User-Agent": "home-energy-ai/1.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        results = response.json()
        return results
    except Exception:
        return []


st.title("⚡ Home Energy Intelligence AI")
st.caption("A simple AI-powered home electricity estimator inspired by modern home energy report tools.")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Enter Home Details")

    address_query = st.text_input(
        "Address",
        placeholder="Start typing an address...",
        key="address_query"
    )

    suggestions = search_address(address_query)

    selected_address = ""
    if suggestions:
        options = [item["display_name"] for item in suggestions]
        selected_address = st.selectbox(
            "Choose a suggested address",
            options,
            key="selected_address"
        )
    elif address_query and len(address_query.strip()) >= 3:
        st.caption("No suggestions found. You can still continue with the typed address.")
        selected_address = address_query

    totrooms = st.number_input("Total Rooms", min_value=1, max_value=20, value=6)
    totsqft_en = st.number_input("Square Footage", min_value=200, max_value=10000, value=1800)
    nhsldmem = st.number_input("Household Members", min_value=1, max_value=15, value=3)

    housing_type_label = st.selectbox(
        "Housing Type",
        ["Single-family detached", "Single-family attached", "Apartment", "Mobile home", "Other"]
    )
    housing_type_map = {
        "Single-family detached": 2,
        "Single-family attached": 3,
        "Apartment": 4,
        "Mobile home": 5,
        "Other": 1
    }
    typehuq = housing_type_map[housing_type_label]

    aircond_label = st.selectbox("Air Conditioning", ["Yes", "No"])
    aircond_map = {"Yes": 1, "No": 0}
    aircond = aircond_map[aircond_label]

    heating_label = st.selectbox(
        "Heating Fuel",
        ["Electricity", "Natural Gas", "Fuel Oil", "Propane", "Other"]
    )
    fuelheat_map = {
        "Electricity": 5,
        "Natural Gas": 1,
        "Fuel Oil": 2,
        "Propane": 3,
        "Other": 10
    }
    fuelheat = fuelheat_map[heating_label]

    division = st.selectbox("Region Division Code", list(range(1, 11)), index=4)
    yearmaderange = st.selectbox("Home Age Range Code", list(range(1, 10)), index=4)

    predict = st.button("Predict Usage", use_container_width=True)

with right:
    st.subheader("Energy Report")

    if predict:
        input_df = pd.DataFrame([{
            "TOTROOMS": totrooms,
            "TOTSQFT_EN": totsqft_en,
            "TYPEHUQ": typehuq,
            "NHSLDMEM": nhsldmem,
            "AIRCOND": aircond,
            "FUELHEAT": fuelheat,
            "DIVISION": division,
            "YEARMADERANGE": yearmaderange
        }])

        prediction = model.predict(input_df)[0]
        monthly_estimate = prediction / 12

        if prediction < 6000:
            category = "Low"
        elif prediction < 12000:
            category = "Medium"
        else:
            category = "High"

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Usage", f"{prediction:,.0f} kWh")
        c2.metric("Monthly Estimate", f"{monthly_estimate:,.0f} kWh")
        c3.metric("Usage Level", category)

        st.markdown("### Why this estimate?")
        st.write(
            "The estimate is mainly driven by home size, heating profile, number of rooms, "
            "household size, housing type, and region."
        )

        st.markdown("### Suggested Next Steps")
        if prediction >= 12000:
            st.warning("High estimated usage: review HVAC efficiency, insulation, and appliance usage.")
        elif prediction >= 6000:
            st.info("Moderate estimated usage: a smart thermostat and efficient appliances may help.")
        else:
            st.success("Lower estimated usage: your home appears relatively efficient based on the inputs.")

        if selected_address:
            st.caption(f"Report generated for: {selected_address}")
        elif address_query:
            st.caption(f"Report generated for: {address_query}")
    else:
        st.write("Fill in the home details and click **Predict Usage** to generate your energy report.")

st.markdown("---")
st.caption("Built with Streamlit and a Random Forest model trained on RECS household energy data.")