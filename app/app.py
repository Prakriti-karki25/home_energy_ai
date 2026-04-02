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

st.markdown("""
<style>
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }

    .sub-title {
        font-size: 1.1rem;
        color: #b8c2d9;
        margin-bottom: 1.5rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        min-height: 120px;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #9fb0d0;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
    }

    .report-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .highlight-box {
        background: linear-gradient(90deg, #5a6f1f, #6e8328);
        color: white;
        padding: 1rem;
        border-radius: 14px;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .small-muted {
        color: #b8c2d9;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "energy_model.pkl"


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


# HERO SECTION
st.markdown('<div class="main-title">⚡ Know your home’s energy story instantly</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Get an AI-powered estimate of annual energy use, monthly usage, and practical next-step recommendations for your home.</div>',
    unsafe_allow_html=True
)

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <b>⚡ Fast estimate</b><br><br>
        See annual and monthly usage in seconds based on your home profile.
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <b>🏠 Personalized inputs</b><br><br>
        Adjust rooms, square footage, home type, heating source, and household size.
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <b>💡 Smart recommendations</b><br><br>
        Get simple next steps based on your estimated usage level.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">Enter Home Details</div>', unsafe_allow_html=True)

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

    with st.expander("Advanced home details", expanded=True):
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

    predict = st.button("Generate Energy Report", use_container_width=True)

with right:
    st.markdown('<div class="section-title">Energy Report</div>', unsafe_allow_html=True)

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
            recommendation_text = "Lower estimated usage: your home appears relatively efficient based on the inputs."
        elif prediction < 12000:
            category = "Medium"
            recommendation_text = "Moderate estimated usage: a smart thermostat and efficient appliances may help."
        else:
            category = "High"
            recommendation_text = "High estimated usage: review HVAC efficiency, insulation, and major appliance usage."

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annual Usage</div>
                <div class="metric-value">{prediction:,.0f} kWh</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Monthly Estimate</div>
                <div class="metric-value">{monthly_estimate:,.0f} kWh</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Usage Level</div>
                <div class="metric-value">{category}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown("### Why this estimate?")
        st.write(
            "This estimate is mainly driven by home size, heating profile, number of rooms, "
            "household size, housing type, and region."
        )

        st.markdown("### Suggested Next Steps")
        st.markdown(
            f'<div class="highlight-box">{recommendation_text}</div>',
            unsafe_allow_html=True
        )

        if selected_address:
            st.markdown(f"<p class='small-muted'>Report generated for: {selected_address}</p>", unsafe_allow_html=True)
        elif address_query:
            st.markdown(f"<p class='small-muted'>Report generated for: {address_query}</p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="report-box">
            <p class="small-muted">
                Fill in the home details and click <b>Generate Energy Report</b> to see your estimate.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit and a Random Forest model trained on RECS household energy data.")