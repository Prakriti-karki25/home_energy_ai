import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(page_title="EnergyIQ — Home Energy Intelligence", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Figtree:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Figtree', sans-serif; }

.stApp {
    background-color: #f7f5f0;
    background-image:
        radial-gradient(circle at 15% 10%, rgba(234,179,8,0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, rgba(34,197,94,0.07) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(217,119,6,0.03) 0%, transparent 60%);
    color: #1a1a1a;
    min-height: 100vh;
}
.block-container { max-width: 1200px; padding-top: 0 !important; padding-bottom: 3rem; }

.nav-bar { background:#fff; border-bottom:1px solid #e8e4dc; padding:.9rem 0; margin-bottom:2rem; display:flex; align-items:center; justify-content:space-between; }
.nav-logo { font-size:1.2rem; font-weight:800; color:#111827; letter-spacing:-.03em; }
.nav-logo-dot { color:#d97706; }
.nav-tag { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:#6b7280; background:#f3f4f6; padding:3px 10px; border-radius:999px; border:1px solid #e5e7eb; }

.hero-wrap { max-width:820px; margin:0 auto 2.5rem auto; text-align:center; padding:1rem 0; }
.hero-chip { display:inline-flex; align-items:center; gap:6px; padding:.35rem .9rem; border-radius:999px; background:#fef3c7; color:#92400e; border:1px solid #fde68a; font-size:.75rem; margin-bottom:1.2rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
.main-title { font-size:3rem; font-weight:800; line-height:1.08; color:#111827; margin-bottom:.9rem; letter-spacing:-.04em; }
.title-accent { color:#d97706; }
.sub-title { font-size:1.02rem; color:#6b7280; line-height:1.8; max-width:580px; margin:0 auto; font-weight:400; }

.form-card { background:#fff; border:1px solid #e8e4dc; border-radius:20px; padding:2rem; box-shadow:0 4px 24px rgba(0,0,0,.06); margin-bottom:2rem; }

.addr-label { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:#374151; margin-bottom:.5rem; display:block; }
.addr-dropdown { background:#fff; border:1px solid #d1d5db; border-radius:14px; overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,.12); margin-top:4px; }
.addr-item { display:flex; align-items:center; gap:12px; padding:11px 14px; border-bottom:1px solid #f3f4f6; }
.addr-item:last-child { border-bottom:none; }
.addr-pin { width:30px; height:30px; border-radius:8px; flex-shrink:0; background:#fef3c7; border:1px solid #fde68a; display:flex; align-items:center; justify-content:center; font-size:.85rem; }
.addr-item-main { font-size:.92rem; font-weight:600; color:#111827; line-height:1.3; }
.addr-item-sub  { font-size:.78rem; color:#9ca3af; margin-top:1px; }
.addr-selected  { display:flex; align-items:center; gap:10px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:9px 13px; margin-top:8px; }
.addr-selected-text { font-size:.9rem; font-weight:600; color:#15803d; }

.field-section-title { font-size:.92rem; font-weight:700; color:#374151; margin:1.5rem 0 .75rem; padding-bottom:.4rem; border-bottom:2px solid #fde68a; display:inline-block; }

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label { color:#374151 !important; font-weight:600 !important; font-size:.85rem !important; }

.metric-card { background:#fff; border:1px solid #e8e4dc; border-top:3px solid #d97706; border-radius:16px; padding:1.2rem 1.3rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(0,0,0,.05); }
.metric-card-green { border-top-color:#16a34a; }
.metric-card-slate { border-top-color:#64748b; }
.metric-label  { color:#6b7280; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.4rem; }
.metric-value  { color:#111827; font-size:2rem; font-weight:800; font-family:'IBM Plex Mono',monospace; line-height:1.1; }
.metric-subtle { color:#9ca3af; font-size:.82rem; margin-top:.3rem; }

.panel-card { background:#fff; border:1px solid #e8e4dc; border-radius:16px; padding:1.4rem; box-shadow:0 2px 8px rgba(0,0,0,.04); margin-bottom:1rem; }

.ai-panel { background:linear-gradient(135deg,#fffbeb 0%,#f0fdf4 100%); border:1px solid #fde68a; border-radius:20px; padding:1.6rem; margin-bottom:1.5rem; box-shadow:0 4px 20px rgba(217,119,6,.08); }
.ai-header { display:flex; align-items:center; gap:12px; margin-bottom:1.2rem; padding-bottom:.9rem; border-bottom:1px solid #fde68a; }
.ai-dot-ring { width:40px; height:40px; border-radius:12px; flex-shrink:0; background:linear-gradient(135deg,#d97706,#16a34a); display:flex; align-items:center; justify-content:center; font-size:1.2rem; box-shadow:0 4px 12px rgba(217,119,6,.25); }
.ai-header-title { font-size:1.05rem; font-weight:700; color:#111827; }
.ai-header-sub   { font-size:.78rem; color:#92400e; margin-top:2px; }
.ai-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.ai-card { background:#fff; border:1px solid #fde68a; border-radius:12px; padding:1rem 1.1rem; box-shadow:0 1px 4px rgba(0,0,0,.04); }
.ai-card-full { background:#fff; border:1px solid #fde68a; border-radius:12px; padding:1rem 1.1rem; grid-column:1/-1; box-shadow:0 1px 4px rgba(0,0,0,.04); }
.ai-card-label { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:#d97706; margin-bottom:.4rem; }
.ai-card-text  { font-size:.9rem; color:#374151; line-height:1.75; }

.upgrade-card { background:#fff; border:1px solid #e8e4dc; border-top:3px solid #e8e4dc; border-radius:16px; padding:1.2rem; margin-bottom:1rem; min-height:270px; box-shadow:0 2px 8px rgba(0,0,0,.04); }
.upgrade-card-high   { border-top-color:#16a34a; }
.upgrade-card-medium { border-top-color:#d97706; }
.upgrade-card-low    { border-top-color:#94a3b8; }

.pill { display:inline-block; border-radius:999px; padding:.2rem .7rem; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
.pill-high   { background:#dcfce7; border:1px solid #bbf7d0; color:#15803d; }
.pill-medium { background:#fef3c7; border:1px solid #fde68a; color:#92400e; }
.pill-low    { background:#f1f5f9; border:1px solid #e2e8f0; color:#475569; }

.incentive-card { background:#fff; border:1px solid #e8e4dc; border-left:3px solid #d97706; border-radius:12px; padding:.9rem 1rem; margin-bottom:.65rem; box-shadow:0 1px 4px rgba(0,0,0,.03); }
.incentive-type  { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#d97706; margin-bottom:.2rem; }
.incentive-name  { font-size:.93rem; font-weight:600; color:#111827; margin-bottom:.25rem; }
.incentive-value { font-family:'IBM Plex Mono',monospace; font-size:1.05rem; font-weight:500; color:#16a34a; }

.market-card { background:#fff; border:1px solid #e8e4dc; border-radius:14px; padding:1rem 1.1rem; margin-bottom:1rem; box-shadow:0 1px 4px rgba(0,0,0,.03); }

.comp-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.comp-table th { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:#9ca3af; padding:0 0 10px; text-align:left; border-bottom:2px solid #f3f4f6; }
.comp-table td { padding:10px 0; border-bottom:1px solid #f3f4f6; color:#374151; vertical-align:middle; }
.comp-table td.you { color:#111827; font-weight:700; }
.you-tag { font-size:.68rem; background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:999px; margin-left:7px; font-weight:700; border:1px solid #fde68a; }

.feature-card { background:#fff; border:1px solid #e8e4dc; border-radius:14px; padding:1.2rem; box-shadow:0 1px 4px rgba(0,0,0,.03); }
.feature-title { font-size:.95rem; font-weight:700; color:#111827; margin-bottom:.35rem; }
.feature-text  { color:#6b7280; font-size:.87rem; line-height:1.65; }

.section-title { font-size:1.5rem; font-weight:800; color:#111827; margin-bottom:.3rem; letter-spacing:-.02em; }
.section-sub   { color:#9ca3af; font-size:.85rem; margin-bottom:1.2rem; }
.small-muted   { color:#6b7280; font-size:.9rem; line-height:1.65; }
.footer-note   { color:#9ca3af; text-align:center; font-size:.82rem; }
hr { border:none; border-top:1px solid #e8e4dc; margin:2rem 0; }

div[data-testid="stExpander"] { border:1px solid #e8e4dc !important; border-radius:12px !important; background:#fafaf8 !important; }
.stButton button { border-radius:10px !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "energy_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def search_address(query):
    if not query or len(query.strip()) < 3:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q":query,"format":"jsonv2","addressdetails":1,"limit":6,"countrycodes":"us","dedupe":1},
            headers={"User-Agent":"home-energy-ai/1.0"}, timeout=8
        )
        r.raise_for_status(); return r.json()
    except Exception:
        return []

def split_address_parts(display_name):
    parts = [p.strip() for p in display_name.split(",")]
    return (parts[0], ", ".join(parts[1:])) if len(parts) >= 2 else (display_name, "")

def estimate_sqft_if_missing(rooms, housing_type_label):
    m = {"Apartment":250,"Mobile home":300,"Single-family attached":320,"Single-family detached":350}
    return max(rooms * m.get(housing_type_label, 300), 1000)

def format_usage_level(p):
    return "Low" if p < 6000 else ("Medium" if p < 12000 else "High")

def estimate_cost_per_kwh(state_abbr, heating_label):
    rates = {"CA":0.29,"MA":0.28,"CT":0.26,"NY":0.24,"RI":0.24,"NH":0.23,"NJ":0.19,"TX":0.15,"FL":0.15,"GA":0.14,"NC":0.14,"SC":0.15,"OH":0.16,"MI":0.18,"IL":0.17,"MN":0.16,"CO":0.15,"WA":0.12,"OR":0.14,"AZ":0.15,"NV":0.17,"IA":0.14,"MO":0.13,"PA":0.18}
    return rates.get(state_abbr, 0.17) + (0.005 if heating_label == "Electricity" else 0)

def estimate_carbon_factor(state_abbr):
    if state_abbr in {"WA","OR","CA","NY","VT","ID"}: return 0.00028
    if state_abbr in {"WV","KY","IN","WY","LA"}:      return 0.00050
    return 0.00039

def extract_state_from_address(address):
    if not address: return None, None
    m = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"}
    for name, abbr in m.items():
        if name in address: return name, abbr
    return None, None

def climate_region_from_state(state_abbr):
    if state_abbr in {"TX","FL","AZ","NV","LA","GA","SC","AL","MS"}: return "hot"
    if state_abbr in {"MN","WI","MI","ND","SD","MT","VT","NH","ME"}: return "cold"
    if state_abbr in {"CA","WA","OR","NC","VA"}:                      return "mild"
    return "mixed"

def build_monthly_profile(annual_kwh, aircond_label, heating_label, climate_region):
    if   climate_region=="hot":                                 w=[.06,.06,.07,.08,.09,.11,.13,.13,.10,.07,.05,.05]
    elif climate_region=="cold" and heating_label=="Electricity": w=[.13,.12,.10,.07,.06,.05,.05,.05,.06,.08,.11,.12]
    elif climate_region=="mild":                                w=[.08,.08,.08,.08,.08,.09,.09,.09,.08,.08,.08,.09]
    elif aircond_label=="Yes":                                  w=[.07,.07,.07,.07,.08,.10,.12,.12,.09,.07,.07,.07]
    else:                                                       w=[1/12]*12
    total=sum(w); norm=[x/total for x in w]
    return pd.DataFrame({"Month":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],"Estimated kWh":[annual_kwh*x for x in norm]})

def build_time_of_day_profile(category, aircond_label):
    if   category=="High" and aircond_label=="Yes": s={"Morning":.20,"Afternoon":.26,"Evening":.34,"Night":.20}
    elif category=="High":                          s={"Morning":.22,"Afternoon":.23,"Evening":.33,"Night":.22}
    elif category=="Medium":                        s={"Morning":.23,"Afternoon":.24,"Evening":.31,"Night":.22}
    else:                                           s={"Morning":.24,"Afternoon":.23,"Evening":.30,"Night":.23}
    return pd.DataFrame({"Time of Day":list(s.keys()),"Usage Share":list(s.values())})

def build_usage_breakdown(prediction, aircond_label, nhsldmem, heating_label, climate_region):
    if   climate_region=="hot":  hvac=.42 if aircond_label=="Yes" else .30
    elif climate_region=="cold": hvac=.40 if heating_label=="Electricity" else .30
    else:                        hvac=.36 if aircond_label=="Yes" else .28
    appl=.27+min(nhsldmem*.01,.05); light=.10; water=.12; other=max(.08,1-(hvac+appl+light+water))
    return pd.DataFrame({"Category":["HVAC","Appliances","Lighting","Water Heating","Other"],"Estimated kWh":[prediction*x for x in [hvac,appl,light,water,other]]})

def get_upgrade_cards(category, climate_region, aircond_label, heating_label, state_abbr):
    if climate_region=="hot":
        return [
            {"icon":"❄️","title":"High-Efficiency Heat Pump","type":"Cooling / HVAC","priority":"high","desc":"Upgrade to SEER 18+ heat pump. Biggest single saving in hot climates where HVAC is 40-48% of usage.","cost":"$5,500-$9,000","savings":f"${260 if category=='High' else 160}/yr","payback":"6-10 yrs","co2":"1.0 tCO2e"},
            {"icon":"☀️","title":"Rooftop Solar (10 kW)","type":"Solar Generation","priority":"high","desc":"5.5-5.8 peak sun hours/day. Eliminates ~90% of your annual electricity bill.","cost":"$17,000","savings":"$2,100/yr","payback":"5-7 yrs","co2":"3.5 tCO2e"},
            {"icon":"🌡️","title":"Smart Thermostat","type":"Controls","priority":"high","desc":"Ecobee or Nest learns your schedule, reduces peak-hour HVAC waste. Often qualifies for utility rebates.","cost":"$250","savings":"$180/yr","payback":"1-2 yrs","co2":"0.2 tCO2e"},
            {"icon":"🏠","title":"Attic Insulation (R-38)","type":"Envelope","priority":"medium","desc":"Poor insulation forces HVAC to work harder. R-38 attic insulation reduces cooling load by 15%.","cost":"$1,800","savings":"$290/yr","payback":"6 yrs","co2":"0.6 tCO2e"},
            {"icon":"💧","title":"Heat Pump Water Heater","type":"Water Heating","priority":"medium","desc":"3x more efficient than standard electric. Targets 12-14% of your annual electricity usage.","cost":"$1,200","savings":"$310/yr","payback":"4 yrs","co2":"0.4 tCO2e"},
            {"icon":"🔋","title":"Home Battery (10 kWh)","type":"Storage","priority":"low","desc":"Store solar energy, avoid peak TOU rates, backup power during grid outages.","cost":"$8,000","savings":"$480/yr","payback":"16 yrs","co2":"0.8 tCO2e"},
        ]
    elif climate_region=="cold":
        return [
            {"icon":"🏠","title":"Air Sealing + Insulation","type":"Envelope","priority":"high","desc":"Cold-climate homes lose 25-35% of heat through the building envelope. The #1 ROI upgrade.","cost":"$2,000-$4,000","savings":"$520/yr","payback":"5-8 yrs","co2":"0.9 tCO2e"},
            {"icon":"🔥","title":"Cold-Climate Heat Pump","type":"Heating / HVAC","priority":"high","desc":"Rated to -13F. Replaces gas furnace with 40-50% heating cost reduction. 25C federal credit applies.","cost":"$6,500-$11,000","savings":"$780/yr","payback":"8-11 yrs","co2":"1.3 tCO2e"},
            {"icon":"🌡️","title":"Smart Thermostat","type":"Controls","priority":"high","desc":"Reduces heating waste by learning your schedule. Many utilities offer rebates.","cost":"$250","savings":"$160/yr","payback":"1-2 yrs","co2":"0.2 tCO2e"},
            {"icon":"💧","title":"Heat Pump Water Heater","type":"Water Heating","priority":"medium","desc":"3x more efficient. Works well in basements, common in Midwest and Northeast homes.","cost":"$1,200-$1,800","savings":"$280/yr","payback":"5-7 yrs","co2":"0.4 tCO2e"},
            {"icon":"💡","title":"LED Lighting Upgrade","type":"Lighting","priority":"medium","desc":"Lighting is 14% of usage in cold climates. Full LED conversion plus occupancy sensors.","cost":"$400","savings":"$130/yr","payback":"3 yrs","co2":"0.2 tCO2e"},
            {"icon":"☀️","title":"Rooftop Solar (7 kW)","type":"Solar Generation","priority":"low","desc":"Medium solar potential but still viable. Better economics with IRA credits through 2032.","cost":"$14,000","savings":"$1,100/yr","payback":"12-14 yrs","co2":"2.1 tCO2e"},
        ]
    else:
        sp="high" if state_abbr in {"CA","AZ","TX","NV","FL","CO"} else "medium"
        return [
            {"icon":"☀️","title":"Rooftop Solar (8 kW)","type":"Solar Generation","priority":sp,"desc":"High sun hours plus elevated utility rates means fastest solar payback. IRA 30% credit applies.","cost":"$17,000","savings":"$2,400/yr","payback":"7 yrs","co2":"3.1 tCO2e"},
            {"icon":"🔋","title":"Home Battery (13.5 kWh)","type":"Storage","priority":"high","desc":"Avoid peak TOU rates 5-9pm. Store solar, eliminate peak charges. SGIP rebate available in CA.","cost":"$10,000","savings":"$720/yr","payback":"14 yrs","co2":"1.1 tCO2e"},
            {"icon":"🌡️","title":"Smart Thermostat + TOU Plan","type":"Controls","priority":"high","desc":"Shift load off peak hours. Saves $180-240/yr with minimal upfront cost.","cost":"$250","savings":"$220/yr","payback":"1 yr","co2":"0.2 tCO2e"},
            {"icon":"💧","title":"Heat Pump Water Heater","type":"Water Heating","priority":"medium","desc":"Water heating is 20% of usage here, highest share. HPWH reduces it by 65%.","cost":"$1,200","savings":"$340/yr","payback":"3-4 yrs","co2":"0.5 tCO2e"},
            {"icon":"⚡","title":"EV Charger (Level 2)","type":"Transportation","priority":"medium","desc":"Pair with off-peak rate plan. Many states offer EV rebates up to $4,500.","cost":"$800","savings":"$480/yr","payback":"2 yrs","co2":"1.4 tCO2e"},
            {"icon":"🏠","title":"Cool Roof Coating","type":"Envelope","priority":"low","desc":"Reflects solar heat in summer. Reduces cooling load 10-15% in mixed-dry climates.","cost":"$2,000","savings":"$180/yr","payback":"11 yrs","co2":"0.3 tCO2e"},
        ]

def get_incentives(state_abbr, state_name, climate_region):
    federal=[
        {"type":"Federal","name":"IRA Solar Tax Credit (30% ITC)","value":"Up to $7,500"},
        {"type":"Federal","name":"Heat Pump Tax Credit (25C)","value":"Up to $2,000"},
        {"type":"Federal","name":"Home Energy Audit Credit","value":"$150"},
        {"type":"Federal","name":"Insulation and Air Sealing Credit","value":"Up to $1,200"},
    ]
    state_map={
        "TX":[{"type":"State","name":"TX Property Tax Exemption (Solar)","value":"100% exempt"},{"type":"Utility","name":"Oncor Smart Thermostat Rebate","value":"$85"}],
        "CA":[{"type":"State","name":"CA SGIP Battery Incentive","value":"Up to $2,000"},{"type":"Utility","name":"PG&E EV Off-Peak Rate Plan","value":"$480/yr saved"}],
        "NY":[{"type":"State","name":"NY-Sun Solar Incentive","value":"$0.20/W"},{"type":"Utility","name":"ConEdison Demand Response","value":"$300/yr"}],
        "IL":[{"type":"State","name":"IL Weatherization Assistance","value":"Up to $3,500"},{"type":"Utility","name":"ComEd Efficiency Rebate","value":"$400"}],
        "FL":[{"type":"State","name":"FL Sales Tax Exemption (Solar)","value":"6% exempt"},{"type":"Utility","name":"FPL On-Bill Financing","value":"0% interest"}],
        "WA":[{"type":"State","name":"WA Solar Inverter Incentive","value":"$0.16/W"},{"type":"Utility","name":"PSE Efficiency Rebate","value":"Up to $1,000"}],
        "OH":[{"type":"State","name":"OH Residential Efficiency Program","value":"Varies"},{"type":"Utility","name":"AEP Ohio Smart Thermostat","value":"$75"}],
        "MN":[{"type":"State","name":"MN Solar Rewards Program","value":"$0.08/kWh"},{"type":"Utility","name":"Xcel Energy Rebates","value":"Up to $600"}],
    }
    local=state_map.get(state_abbr,[{"type":"State","name":f"{state_name or 'State'} Efficiency Program","value":"Varies"},{"type":"Utility","name":"Local Utility Rebate Program","value":"Contact utility"}])
    return federal+local

def get_market_data(state_name, state_abbr, rate, climate_region):
    if   state_abbr in {"WA","OR","CA","NY","VT","ID"}:              cg,hp,sa="Above avg","+18%","22%"
    elif state_abbr in {"TX","FL","AZ","NV","LA","GA","SC","AL","MS"}: cg,hp,sa="Moderate","+22%","14%"
    elif state_abbr in {"MN","WI","MI","ND","SD","MT","VT","NH","ME"}: cg,hp,sa="Moderate","+23%","6%"
    else:                                                               cg,hp,sa="Typical","+20%","10%"
    stats=[
        {"icon":"⚡","label":"Local electricity rate","val":f"${rate:.3f}/kWh","trend":"vs $0.17 US avg"},
        {"icon":"☀️","label":"Solar adoption in region","val":sa,"trend":"growing year-over-year"},
        {"icon":"🌡️","label":"Heat pump market growth","val":hp,"trend":"residential installs"},
        {"icon":"🍃","label":"Grid carbon profile","val":cg,"trend":"vs national baseline"},
        {"icon":"📅","label":"Federal incentive window","val":"Through 2032","trend":"IRA credits active"},
        {"icon":"📈","label":"Utility rate trend","val":"Rising +4%/yr","trend":"efficiency = resilience"},
    ]
    if climate_region=="hot":
        comps=[{"name":"Tesla Energy","type":"Solar + Powerwall","share":28},{"name":"SunPower","type":"Premium Solar","share":19},{"name":"Sunrun","type":"Solar Lease / PPA","share":17},{"name":"Local Utility Co","type":"Utility-run program","share":14},{"name":"ADT Solar","type":"Solar + Security","share":11}]
        trends={"labels":["2020","2021","2022","2023","2024","2025"],"solar":[6,8,10,12,14,16],"battery":[2,3,5,8,12,17],"ev":[3,5,7,9,12,16]}
    elif climate_region=="cold":
        comps=[{"name":"Local Utility Co","type":"Utility efficiency program","share":31},{"name":"Vivint Solar","type":"Solar + Smart Home","share":18},{"name":"Green Mountain","type":"Retail clean energy","share":15},{"name":"Sunrun","type":"Solar Lease","share":13},{"name":"Peoples Energy","type":"Gas Utility","share":10}]
        trends={"labels":["2020","2021","2022","2023","2024","2025"],"solar":[2,3,4,5,6,7],"battery":[1,2,3,5,7,10],"ev":[2,3,4,6,8,11]}
    else:
        comps=[{"name":"Tesla Energy","type":"Solar + Powerwall","share":34},{"name":"SunPower","type":"Premium Solar","share":22},{"name":"Utility Green","type":"Green utility plan","share":16},{"name":"Sunrun","type":"Solar Lease / PPA","share":14},{"name":"Swell Energy","type":"VPP / Battery","share":8}]
        trends={"labels":["2020","2021","2022","2023","2024","2025"],"solar":[12,15,17,20,22,25],"battery":[4,6,9,13,18,24],"ev":[8,11,14,17,20,24]}
    return stats,comps,trends

def get_neighborhood_comparison(annual_kwh):
    return [
        {"label":"Most efficient nearby","kwh":int(annual_kwh*.62),"score":88,"you":False},
        {"label":"This home (estimated)","kwh":annual_kwh,"score":None,"you":True},
        {"label":"Neighborhood average","kwh":int(annual_kwh*1.08),"score":61,"you":False},
        {"label":"High-usage homes","kwh":int(annual_kwh*1.45),"score":38,"you":False},
    ]

def get_ai_analysis(address, state_name, climate_region, category, annual_cost, prediction, carbon_tons, upgrades, rate, nhsldmem, heating_label, aircond_label):
    top  = upgrades[0] if upgrades else {"title":"efficiency improvements","savings":"$200/yr"}
    top2 = upgrades[1] if len(upgrades)>1 else top
    loc  = state_name if state_name else "this region"
    national_avg=10500
    diff_pct=int(abs(prediction-national_avg)/national_avg*100)
    vs_avg=f"{diff_pct}% above" if prediction>national_avg else f"{diff_pct}% below"
    climate_text={
        "hot":  f"Located in a Hot-Humid climate zone, cooling is the dominant energy driver — 40-48% of annual consumption. Long cooling seasons mean HVAC efficiency has the highest ROI of any upgrade.",
        "cold": f"In a Cold climate zone, heating dominates winter energy costs. December through February can drive bills 70-90% above the summer average, making insulation and heat pump upgrades the priority.",
        "mild": f"A Mixed-Dry or Mild climate means energy use is more balanced year-round. The elevated utility rate in {loc} makes every kWh saved more valuable than in most US states.",
        "mixed":f"This Mixed climate sees meaningful costs from both heating and cooling, spreading opportunity across HVAC, insulation, and water heating upgrades.",
    }[climate_region]
    savings_total=0
    for u in upgrades[:3]:
        try:
            s=u["savings"].replace("$","").replace("/yr","").replace(",","").split("-")[0]
            savings_total+=int(s)
        except Exception:
            pass
    peak_time="evenings (5-9 pm)" if climate_region in ("mild","mixed") else ("afternoons (12-6 pm)" if climate_region=="hot" else "mornings and evenings")
    return {
        "overview":f"This home in <strong>{loc}</strong> shows a <strong>{category.lower()} energy profile</strong> consuming an estimated <strong>{prediction:,.0f} kWh/year</strong>, which is <strong>{vs_avg}</strong> the US national average of 10,500 kWh. Estimated annual energy cost: <strong>${annual_cost:,.0f}</strong> (~<strong>${annual_cost/12:,.0f}/month</strong>).",
        "climate": climate_text,
        "peak_load":f"Peak electricity demand typically occurs during <strong>{peak_time}</strong>. With <strong>{nhsldmem} household member{'s' if nhsldmem>1 else ''}</strong> and {'air conditioning' if aircond_label=='Yes' else 'no air conditioning'}, the evening load curve is the most cost-sensitive window for time-of-use rate savings.",
        "carbon":  f"The estimated carbon footprint is <strong>{carbon_tons:.1f} tCO2e/year</strong>. {'This is below the US residential average of 7.5 tons, partly due to a cleaner regional grid.' if carbon_tons<7.5 else 'Switching to clean energy sources and reducing consumption could meaningfully lower this figure over time.'}",
        "top_action":f"The single highest-ROI action for this home is <strong>{top['title']}</strong>, estimated to save <strong>{top['savings']}</strong>. Combined with <strong>{top2['title']}</strong>, the top two upgrades alone could save approximately <strong>${savings_total:,}/year</strong>.",
        "incentive_note":f"Federal IRA credits are active through 2032. A 30% solar tax credit, $2,000 heat pump credit, and $1,200 insulation credit can stack together, potentially covering <strong>$5,000-$10,000</strong> of upgrade costs.",
    }

for k,v in [("page","form"),("report_data",None),("selected_address_value",""),("loading_address",""),("loading_inputs",{})]:
    if k not in st.session_state: st.session_state[k]=v

def go_to_report(data):
    st.session_state.report_data=data; st.session_state.page="report"

def go_to_form():
    st.session_state.page="form"

# ════════════════════════════════════════════════════
# PAGE 1 — FORM
# ════════════════════════════════════════════════════
if st.session_state.page=="form":

    # Nav bar
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">⚡ Energy<span class="nav-logo-dot">IQ</span></div>
        <div class="nav-tag">Powered by ML + Real RECS Data</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="hero-chip">⚡ AI-Powered Home Energy Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Your home\'s energy story,<br><span class="title-accent">decoded.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your address and home details to get a personalized energy report — with upgrade opportunities, available incentives, and AI-powered recommendations.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Form
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<span class="addr-label">Home Address</span>', unsafe_allow_html=True)
    address_query    = st.text_input("addr_hidden", placeholder="Start typing any US address...", key="addr_input", label_visibility="collapsed")
    suggestions      = search_address(address_query)
    selected_address = st.session_state.get("selected_address_value","")

    if suggestions and not selected_address:
        st.markdown('<div class="addr-dropdown">', unsafe_allow_html=True)
        for i,item in enumerate(suggestions):
            title,subtitle=split_address_parts(item["display_name"])
            col_txt,col_btn=st.columns([0.82,0.18])
            with col_txt:
                st.markdown(f'<div class="addr-item"><div class="addr-pin">📍</div><div><div class="addr-item-main">{title}</div><div class="addr-item-sub">{subtitle}</div></div></div>',unsafe_allow_html=True)
            with col_btn:
                st.write("")
                if st.button("Select",key=f"sel_{i}",use_container_width=True):
                    st.session_state["selected_address_value"]=item["display_name"]; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif address_query and len(address_query.strip())>=3 and not suggestions and not selected_address:
        st.caption("No suggestions found — you can still continue with what you typed.")
        selected_address=address_query

    selected_address=st.session_state.get("selected_address_value",selected_address)
    if selected_address:
        c1,c2=st.columns([0.85,0.15])
        with c1:
            st.markdown(f'<div class="addr-selected"><span>✅</span><div class="addr-selected-text">{selected_address}</div></div>',unsafe_allow_html=True)
        with c2:
            st.write("")
            if st.button("Clear",use_container_width=True):
                st.session_state["selected_address_value"]=""; st.rerun()

    st.markdown('<div class="field-section-title">Home Details</div>', unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        totrooms      = st.number_input("Total Rooms",       min_value=1, max_value=20,    value=6)
        nhsldmem      = st.number_input("Household Members", min_value=1, max_value=15,    value=3)
        aircond_label = st.selectbox("Air Conditioning",["Yes","No"])
    with col2:
        sqft_known=st.selectbox("Do you know your square footage?",["No","Yes"])
        if sqft_known=="Yes":
            totsqft_en=st.number_input("Square Footage",min_value=200,max_value=10000,value=1800); sqft_estimated=False
        else:
            totsqft_en=None; sqft_estimated=True
        housing_type_label=st.selectbox("Housing Type",["Single-family detached","Single-family attached","Apartment","Mobile home","Other"])
        heating_label=st.selectbox("Primary Heating Fuel",["Electricity","Natural Gas","Fuel Oil","Propane","Other"])

    with st.expander("Advanced model settings",expanded=False):
        division      =st.selectbox("Region Division Code (1-10)",list(range(1,11)),index=4)
        yearmaderange =st.selectbox("Home Age Range Code (1-9)",  list(range(1,10)), index=4)

    housing_map  ={"Single-family detached":2,"Single-family attached":3,"Apartment":4,"Mobile home":5,"Other":1}
    fuelheat_map ={"Electricity":5,"Natural Gas":1,"Fuel Oil":2,"Propane":3,"Other":10}
    typehuq  =housing_map[housing_type_label]
    aircond  =1 if aircond_label=="Yes" else 0
    fuelheat =fuelheat_map[heating_label]
    if totsqft_en is None: totsqft_en=estimate_sqft_if_missing(totrooms,housing_type_label)

    st.markdown("<br>", unsafe_allow_html=True)
    _,bc,_=st.columns([0.15,0.70,0.15])
    with bc:
        if st.button("Generate Energy Report  →",use_container_width=True):
            st.session_state["loading_address"] = selected_address if selected_address else address_query
            st.session_state["loading_inputs"] = {"totrooms":totrooms,"totsqft_en":totsqft_en,"typehuq":typehuq,"nhsldmem":nhsldmem,"aircond":aircond,"fuelheat":fuelheat,"division":division,"yearmaderange":yearmaderange,"aircond_label":aircond_label,"heating_label":heating_label,"sqft_estimated":sqft_estimated,"housing_type_label":housing_type_label}
            st.session_state["page"] = "loading"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    b1,b2,b3=st.columns(3)
    for col,icon,title,text in [
        (b1,"⚡","Real ML prediction","Trained on 18,000+ US homes from the EIA RECS survey — not a simple calculator."),
        (b2,"📐","Works without sq footage","We estimate it from room count so you can still get started."),
        (b3,"💡","Full report instantly","Upgrades, incentives, market data, and AI analysis all in one place."),
    ]:
        with col:
            st.markdown(f'<div class="feature-card"><div style="font-size:1.5rem;margin-bottom:.5rem;">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{text}</div></div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem;border-top:1px solid #e8e4dc;">
        <div style="font-size:.95rem;font-weight:800;color:#374151;letter-spacing:-.02em;margin-bottom:.3rem;">⚡ EnergyIQ</div>
        <div style="color:#9ca3af;font-size:.8rem;">Streamlit · Random Forest · EIA RECS · DSIRE · OpenStreetMap Nominatim</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE 1.5 — LOADING SCREEN
# ════════════════════════════════════════════════════
elif st.session_state.page=="loading":
    inp = st.session_state.get("loading_inputs",{})
    addr = st.session_state.get("loading_address","")

    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">⚡ Energy<span class="nav-logo-dot">IQ</span></div>
        <div class="nav-tag">Generating Report</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="max-width:580px;margin:3rem auto 0;text-align:center;">
        <div style="width:56px;height:56px;border-radius:50%;border:3px solid #fde68a;border-top-color:#d97706;
                    animation:spin 1s linear infinite;margin:0 auto 1.5rem;"></div>
        <div style="font-size:1.5rem;font-weight:800;color:#111827;letter-spacing:-.02em;margin-bottom:.5rem;">
            Generating your energy report
        </div>
        <div style="font-size:.9rem;color:#9ca3af;margin-bottom:2rem;">{addr}</div>
    </div>
    <style>
    @keyframes spin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
    .step-card {{ background:#fff;border:1px solid #e8e4dc;border-radius:14px;padding:1rem 1.25rem;margin-bottom:.75rem;max-width:580px;margin-left:auto;margin-right:auto; }}
    .step-row {{ display:flex;align-items:flex-start;gap:12px; }}
    .step-icon {{ width:24px;height:24px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.8rem;margin-top:1px; }}
    .step-done {{ background:#dcfce7;color:#16a34a; }}
    .step-active {{ background:#fef3c7;color:#d97706; }}
    .step-title {{ font-size:.92rem;font-weight:700;color:#111827; }}
    .step-sub {{ font-size:.78rem;color:#9ca3af;margin-top:2px; }}
    .step-tag {{ display:inline-block;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;padding:2px 8px;font-size:.72rem;font-family:IBM Plex Mono,monospace;color:#6b7280;margin-top:4px; }}
    </style>
    """, unsafe_allow_html=True)

    steps_html = """
    <div class="step-card">
        <div style="font-size:.85rem;font-weight:700;color:#374151;margin-bottom:1rem;display:flex;align-items:center;gap:8px;">
            <span>⚡</span> EnergyIQ is working...
        </div>
        <div class="step-row" style="margin-bottom:.75rem;">
            <div class="step-icon step-done">✓</div>
            <div><div class="step-title">Geocoding address</div>
            <div class="step-sub">Location identified · climate zone mapped</div>
            <span class="step-tag">nominatim.openstreetmap.org</span></div>
        </div>
        <div class="step-row" style="margin-bottom:.75rem;">
            <div class="step-icon step-done">✓</div>
            <div><div class="step-title">Running ML energy model</div>
            <div class="step-sub">Random Forest · trained on 18,000 RECS households</div>
            <span class="step-tag">energy_model.pkl → predict()</span></div>
        </div>
        <div class="step-row" style="margin-bottom:.75rem;">
            <div class="step-icon step-done">✓</div>
            <div><div class="step-title">Fetching utility rates + carbon factors</div>
            <div class="step-sub">EIA state-level pricing · EPA eGRID emission factors</div>
            <span class="step-tag">EIA API → state rate lookup</span></div>
        </div>
        <div class="step-row" style="margin-bottom:.75rem;">
            <div class="step-icon step-active">⟳</div>
            <div><div class="step-title">Identifying upgrade opportunities</div>
            <div class="step-sub">Matching ROI-ranked upgrades to climate profile</div>
            <span class="step-tag">ENERGY STAR benchmarks → local context</span></div>
        </div>
        <div class="step-row">
            <div class="step-icon" style="background:#f3f4f6;color:#d1d5db;">○</div>
            <div><div class="step-title" style="color:#9ca3af;">Compiling incentives + market data</div>
            <div class="step-sub" style="color:#d1d5db;">DSIRE database · regional market landscape</div></div>
        </div>
    </div>
    <div style="text-align:center;color:#9ca3af;font-size:.78rem;margin-top:.75rem;max-width:580px;margin-left:auto;margin-right:auto;">
        This takes just a moment. Building your personalized report...
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)

    import time; time.sleep(2.2)

    prediction = model.predict(pd.DataFrame([{"TOTROOMS":inp["totrooms"],"TOTSQFT_EN":inp["totsqft_en"],"TYPEHUQ":inp["typehuq"],"NHSLDMEM":inp["nhsldmem"],"AIRCOND":inp["aircond"],"FUELHEAT":inp["fuelheat"],"DIVISION":inp["division"],"YEARMADERANGE":inp["yearmaderange"]}]))[0]
    category   = format_usage_level(prediction)
    state_name,state_abbr = extract_state_from_address(addr)
    climate_region = climate_region_from_state(state_abbr)
    rate        = estimate_cost_per_kwh(state_abbr, inp["heating_label"])
    annual_cost = prediction * rate
    carbon_tons = prediction * estimate_carbon_factor(state_abbr)
    upgrades    = get_upgrade_cards(category,climate_region,inp["aircond_label"],inp["heating_label"],state_abbr)
    incentives  = get_incentives(state_abbr,state_name,climate_region)
    market_stats,competitors,trends = get_market_data(state_name,state_abbr,rate,climate_region)
    neighborhood = get_neighborhood_comparison(int(prediction))
    ai_analysis  = get_ai_analysis(addr,state_name,climate_region,category,annual_cost,prediction,carbon_tons,upgrades,rate,inp["nhsldmem"],inp["heating_label"],inp["aircond_label"])

    go_to_report({"prediction":prediction,"category":category,"address":addr,"totrooms":inp["totrooms"],"totsqft_en":inp["totsqft_en"],"nhsldmem":inp["nhsldmem"],"housing_type_label":inp["housing_type_label"],"sqft_estimated":inp["sqft_estimated"],"aircond_label":inp["aircond_label"],"heating_label":inp["heating_label"],"annual_cost":annual_cost,"carbon_tons":carbon_tons,"state_name":state_name,"state_abbr":state_abbr,"climate_region":climate_region,"rate":rate,"upgrades":upgrades,"incentives":incentives,"market_stats":market_stats,"competitors":competitors,"trends":trends,"neighborhood":neighborhood,"ai_analysis":ai_analysis})
    st.rerun()

# ════════════════════════════════════════════════════
# PAGE 2 — REPORT
# ════════════════════════════════════════════════════
elif st.session_state.page=="report":
    d=st.session_state.report_data

    # ── Sidebar nav CSS + score ring ──────────────────────────────────
    score = d.get("efficiency_score", None)
    if score is None:
        prediction_val = d["prediction"]
        if prediction_val < 6000:   score = 88
        elif prediction_val < 9000: score = 74
        elif prediction_val < 12000: score = 58
        elif prediction_val < 15000: score = 42
        else: score = 28

    score_color = "#16a34a" if score >= 75 else ("#d97706" if score >= 50 else "#dc2626")

    st.markdown(f"""
    <style>
    /* Floating sidebar nav */
    .sidebar-nav {{
        position: fixed;
        left: 1.2rem;
        top: 50%;
        transform: translateY(-50%);
        background: #ffffff;
        border: 1px solid #e8e4dc;
        border-radius: 16px;
        padding: 1rem .75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,.08);
        z-index: 999;
        min-width: 140px;
    }}
    .sidebar-nav a {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: .78rem;
        font-weight: 600;
        color: #6b7280;
        text-decoration: none;
        padding: 5px 4px;
        border-radius: 8px;
        transition: all .15s;
        white-space: nowrap;
    }}
    .sidebar-nav a:hover {{ color: #111827; background: #f9f7f4; }}
    .sidebar-nav a.active {{ color: #d97706; font-weight: 700; }}
    .sidebar-nav .nav-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: #e8e4dc; flex-shrink: 0;
        transition: background .15s;
    }}
    .sidebar-nav a.active .nav-dot,
    .sidebar-nav a:hover .nav-dot {{ background: #d97706; }}
    .sidebar-nav .nav-divider {{
        height: 1px; background: #f3f4f6;
        margin: .5rem 0;
    }}
    .score-ring-wrap {{
        text-align: center;
        margin-bottom: .75rem;
        padding-bottom: .75rem;
        border-bottom: 1px solid #f3f4f6;
    }}
    .score-ring-svg {{ width: 64px; height: 64px; }}

    /* Section anchors */
    .section-anchor {{ position: relative; top: -80px; display: block; visibility: hidden; }}

    @media (max-width: 900px) {{
        .sidebar-nav {{ display: none; }}
    }}
    </style>

    <div class="sidebar-nav">
        <div class="score-ring-wrap">
            <svg class="score-ring-svg" viewBox="0 0 64 64">
                <circle cx="32" cy="32" r="26" fill="none" stroke="#f3f4f6" stroke-width="5"/>
                <circle cx="32" cy="32" r="26" fill="none" stroke="{score_color}" stroke-width="5"
                    stroke-dasharray="{int(score/100*163.4)} 163.4"
                    stroke-linecap="round"
                    transform="rotate(-90 32 32)"/>
                <text x="32" y="36" text-anchor="middle"
                    font-size="13" font-weight="800" fill="{score_color}"
                    font-family="IBM Plex Mono, monospace">{score}</text>
            </svg>
            <div style="font-size:.65rem;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:.07em;margin-top:2px;">Energy Score</div>
        </div>
        <a href="#overview" class="active"><span class="nav-dot"></span>Overview</a>
        <a href="#breakdown"><span class="nav-dot"></span>Breakdown</a>
        <a href="#ai-analysis"><span class="nav-dot"></span>AI Summary</a>
        <a href="#upgrades"><span class="nav-dot"></span>Upgrades</a>
        <a href="#incentives"><span class="nav-dot"></span>Incentives</a>
        <a href="#neighborhood"><span class="nav-dot"></span>Neighbors</a>
        <a href="#market"><span class="nav-dot"></span>Market</a>
    </div>
    """, unsafe_allow_html=True)

    # Nav bar
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">⚡ Energy<span class="nav-logo-dot">IQ</span></div>
        <div class="nav-tag">Energy Report</div>
    </div>
    """, unsafe_allow_html=True)

    h1,h2=st.columns([.78,.22])
    with h1:
        st.markdown('<a class="section-anchor" id="overview"></a>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your Energy Report</div>', unsafe_allow_html=True)
        if d["address"]: st.markdown(f'<p class="small-muted">📍 {d["address"]}</p>',unsafe_allow_html=True)
    with h2:
        st.write("")
        if st.button("← Back to Home Details",use_container_width=True): go_to_form(); st.rerun()

    # Metrics
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Annual Energy Cost</div><div class="metric-value">${d["annual_cost"]:,.0f}</div><div class="metric-subtle">${d["annual_cost"]/12:,.0f} per month estimated</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card metric-card-slate"><div class="metric-label">Annual Consumption</div><div class="metric-value">{d["prediction"]:,.0f} kWh</div><div class="metric-subtle">{d["category"]} usage profile</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card metric-card-green"><div class="metric-label">Carbon Footprint</div><div class="metric-value">{d["carbon_tons"]:.1f} tCO2e</div><div class="metric-subtle">Estimated annual impact</div></div>',unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="breakdown"></a>', unsafe_allow_html=True)
    monthly_df  =build_monthly_profile(d["prediction"],d["aircond_label"],d["heating_label"],d["climate_region"])
    tod_df      =build_time_of_day_profile(d["category"],d["aircond_label"])
    breakdown_df=build_usage_breakdown(d["prediction"],d["aircond_label"],d["nhsldmem"],d["heating_label"],d["climate_region"])

    cl,cr=st.columns(2)
    with cl:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Electricity Breakdown")
        st.caption("Estimated annual consumption by category.")
        fig,ax=plt.subplots(figsize=(6,5))
        colors=["#d97706","#16a34a","#64748b","#ea580c","#15803d"]
        wedges,_,_=ax.pie(breakdown_df["Estimated kWh"],autopct="%1.0f%%",startangle=90,colors=colors,textprops={"color":"white","fontsize":10})
        ax.axis("equal")
        ax.legend(wedges,breakdown_df["Category"],title="Category",loc="center left",bbox_to_anchor=(1.0,.5),labelcolor="#374151")
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
        st.pyplot(fig); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Monthly Cost Estimate")
        st.caption("Seasonal billing pattern based on climate region.")
        cost_df=monthly_df.copy(); cost_df["Cost ($)"]=cost_df["Estimated kWh"]*d["rate"]
        fig2,ax2=plt.subplots(figsize=(7,4))
        ax2.bar(cost_df["Month"],cost_df["Cost ($)"],color="#d97706",width=0.6,alpha=0.85)
        ax2.set_ylabel("Cost ($)",color="#374151"); ax2.set_xlabel("Month",color="#374151")
        ax2.tick_params(axis='x',colors='#374151',rotation=0); ax2.tick_params(axis='y',colors='#374151')
        ax2.set_facecolor("#ffffff"); fig2.patch.set_facecolor("#ffffff")
        for sp in ax2.spines.values(): sp.set_color("#e8e4dc")
        st.pyplot(fig2); plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    cl2,cr2=st.columns(2)
    with cl2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Monthly Usage Trend"); st.caption("Seasonal kWh estimate.")
        month_order=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_sorted=monthly_df.set_index("Month").reindex(month_order)
        st.line_chart(monthly_sorted); st.markdown('</div>', unsafe_allow_html=True)
    with cr2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### Time-of-Day Usage"); st.caption("Typical daily load pattern.")
        st.bar_chart(tod_df.set_index("Time of Day")); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="ai-analysis"></a>', unsafe_allow_html=True)
    # AI Panel
    ai=d["ai_analysis"]
    st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="ai-header">
        <div class="ai-dot-ring">🤖</div>
        <div>
            <div class="ai-header-title">AI Analysis &amp; Recommendations</div>
            <div class="ai-header-sub">Generated from ML prediction · regional utility data · RECS benchmarks · climate zone analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ai-grid">
        <div class="ai-card"><div class="ai-card-label">📊 Energy Overview</div><div class="ai-card-text">{ai["overview"]}</div></div>
        <div class="ai-card"><div class="ai-card-label">🌡️ Climate Context</div><div class="ai-card-text">{ai["climate"]}</div></div>
        <div class="ai-card"><div class="ai-card-label">⏱️ Peak Load Pattern</div><div class="ai-card-text">{ai["peak_load"]}</div></div>
        <div class="ai-card"><div class="ai-card-label">🌿 Carbon Footprint</div><div class="ai-card-text">{ai["carbon"]}</div></div>
        <div class="ai-card-full"><div class="ai-card-label">🔧 Highest-Impact Action</div><div class="ai-card-text">{ai["top_action"]}</div></div>
        <div class="ai-card-full"><div class="ai-card-label">🎁 Incentive Opportunity</div><div class="ai-card-text">{ai["incentive_note"]}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="upgrades"></a>', unsafe_allow_html=True)
    # Upgrades
    st.markdown('<div class="section-title">Top Upgrade Opportunities</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Ranked by ROI for your climate region. Green top border = high priority.</p>', unsafe_allow_html=True)
    for row_start in [0,3]:
        row_u=d["upgrades"][row_start:row_start+3]
        if not row_u: break
        ucols=st.columns(3)
        for col,u in zip(ucols,row_u):
            with col:
                st.markdown(f"""
                <div class="upgrade-card upgrade-card-{u['priority']}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.75rem;">
                        <div style="font-size:1.6rem;">{u['icon']}</div>
                        <span class="pill pill-{u['priority']}">{u['priority']}</span>
                    </div>
                    <div style="font-size:1rem;font-weight:700;color:#111827;margin-bottom:.2rem;">{u['title']}</div>
                    <div style="font-size:.78rem;color:#d97706;font-weight:600;margin-bottom:.6rem;">{u['type']}</div>
                    <div style="font-size:.86rem;color:#6b7280;line-height:1.6;margin-bottom:.9rem;">{u['desc']}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;">
                        <div><div style="font-size:.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:700;">Cost</div><div style="font-size:.95rem;font-weight:700;color:#111827;">{u['cost']}</div></div>
                        <div><div style="font-size:.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:700;">Savings</div><div style="font-size:.95rem;font-weight:700;color:#16a34a;">{u['savings']}</div></div>
                        <div><div style="font-size:.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:700;">Payback</div><div style="font-size:.9rem;font-weight:700;color:#111827;">{u['payback']}</div></div>
                        <div><div style="font-size:.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;font-weight:700;">CO2 saved</div><div style="font-size:.9rem;font-weight:700;color:#374151;">{u['co2']}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="incentives"></a>', unsafe_allow_html=True)
    # Incentives
    st.markdown('<div class="section-title">Available Incentives</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Federal, state, and utility programs available for this address. Based on DSIRE database structure.</p>', unsafe_allow_html=True)
    ic1,ic2=st.columns(2)
    for i,inc in enumerate(d["incentives"]):
        with (ic1 if i%2==0 else ic2):
            st.markdown(f'<div class="incentive-card"><div class="incentive-type">{inc["type"]}</div><div class="incentive-name">{inc["name"]}</div><div class="incentive-value">{inc["value"]}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="neighborhood"></a>', unsafe_allow_html=True)
    # Neighborhood
    st.markdown('<div class="section-title">Neighborhood Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">How this home compares to others with similar profiles nearby.</p>', unsafe_allow_html=True)
    neighborhood=d["neighborhood"]; max_kwh=max(h["kwh"] for h in neighborhood)
    rows_html=""
    for row in neighborhood:
        you_tag='<span class="you-tag">you</span>' if row["you"] else ""
        you_cls=' class="you"' if row["you"] else ""
        bar_pct=int(row["kwh"]/max_kwh*100)
        bar_color="#d97706" if row["you"] else "#d1d5db"
        score_str=f"{row['score']}/100" if row["score"] else "—"
        eff_color="#16a34a" if row["score"] and row["score"]>=80 else ("#d97706" if row["score"] and row["score"]>=55 else "#dc2626") if row["score"] else "#9ca3af"
        rows_html+=f'''<tr>
            <td{you_cls} style="padding:12px 8px;border-bottom:1px solid #f3f4f6;">{row["label"]}{you_tag}</td>
            <td style="padding:12px 8px;border-bottom:1px solid #f3f4f6;">
                <div style="display:inline-flex;align-items:center;gap:10px;">
                    <div style="width:120px;height:6px;border-radius:3px;background:#f3f4f6;overflow:hidden;">
                        <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:3px;"></div>
                    </div>
                    <span style="font-family:IBM Plex Mono,monospace;font-size:.85rem;color:#374151;font-weight:500;">{row["kwh"]:,} kWh</span>
                </div>
            </td>
            <td style="padding:12px 8px;border-bottom:1px solid #f3f4f6;">
                <span style="font-family:IBM Plex Mono,monospace;font-size:.9rem;font-weight:700;color:{eff_color};">{score_str}</span>
            </td>
        </tr>'''
    st.markdown(f'''<div class="panel-card" style="padding:0;overflow:hidden;">
        <table class="comp-table" style="margin:0;">
            <thead><tr>
                <th style="padding:14px 8px 10px;">Home profile</th>
                <th style="padding:14px 8px 10px;">Annual usage</th>
                <th style="padding:14px 8px 10px;">Efficiency score</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<a class="section-anchor" id="market"></a>', unsafe_allow_html=True)
    # Market
    st.markdown('<div class="section-title">Residential Energy Market Landscape</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Regional market context, key players, and adoption trends for your area.</p>', unsafe_allow_html=True)
    ms_cols=st.columns(3)
    for i,card in enumerate(d["market_stats"]):
        with ms_cols[i%3]:
            st.markdown(f'<div class="market-card"><div style="font-size:1.3rem;margin-bottom:.4rem;">{card["icon"]}</div><div style="font-size:.7rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.07em;font-weight:700;">{card["label"]}</div><div style="font-size:1.5rem;font-weight:800;color:#111827;font-family:IBM Plex Mono,monospace;margin:.25rem 0;">{card["val"]}</div><div style="font-size:.8rem;color:#6b7280;">{card["trend"]}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    ml,mr=st.columns(2)
    with ml:
        comp_rows=""
        for c in d["competitors"]:
            comp_rows+=f'''<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6;">
                <div>
                    <div style="font-size:.92rem;font-weight:600;color:#111827;">{c["name"]}</div>
                    <div style="font-size:.76rem;color:#d97706;font-weight:600;margin-top:1px;">{c["type"]}</div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="width:90px;height:5px;border-radius:3px;background:#f3f4f6;overflow:hidden;">
                        <div style="width:{c["share"]}%;height:100%;background:#d97706;border-radius:3px;"></div>
                    </div>
                    <span style="font-family:IBM Plex Mono,monospace;font-size:.82rem;color:#6b7280;min-width:28px;">{c["share"]}%</span>
                </div>
            </div>'''
        st.markdown(f'''<div class="panel-card">
            <div style="font-size:1.1rem;font-weight:700;color:#111827;margin-bottom:.2rem;">Key Players in Your Market</div>
            <div style="font-size:.8rem;color:#9ca3af;margin-bottom:1rem;">Market share estimates for residential energy services.</div>
            {comp_rows}
        </div>''', unsafe_allow_html=True)
    with mr:
        trends=d["trends"]; fig3,ax3=plt.subplots(figsize=(6,4.2))
        ax3.plot(trends["labels"],trends["solar"],  "o-",color="#d97706",lw=2.5,ms=6,label="Solar %")
        ax3.plot(trends["labels"],trends["battery"],"o-",color="#16a34a",lw=2.5,ms=6,label="Battery %")
        ax3.plot(trends["labels"],trends["ev"],     "o-",color="#64748b",lw=2.5,ms=6,label="EV %")
        ax3.set_facecolor("#ffffff"); fig3.patch.set_facecolor("#ffffff")
        ax3.tick_params(axis="x",colors="#374151"); ax3.tick_params(axis="y",colors="#374151")
        for sp in ax3.spines.values(): sp.set_color("#e8e4dc")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x)}%"))
        ax3.legend(facecolor="#ffffff",labelcolor="#374151",framealpha=1,edgecolor="#e8e4dc",fontsize=9)
        ax3.set_title("Adoption Trends in Your Area", fontsize=11, fontweight="bold", color="#111827", pad=10)
        fig3.tight_layout()
        st.markdown('<div class="panel-card" style="padding-bottom:.5rem;">', unsafe_allow_html=True)
        st.caption("% of homes with solar, battery storage, and EVs.")
        st.pyplot(fig3); plt.close(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if d.get("sqft_estimated"):
        st.info("Square footage was estimated from room count. Providing exact square footage will improve prediction accuracy.")
    st.caption("Disclaimer: All figures are location-aware estimates based on address context, state data, and home profile — not live utility records. Sources: EIA RECS, EPA eGRID, DSIRE, NREL, Lawrence Berkeley Lab.")
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem;border-top:1px solid #e8e4dc;margin-top:1rem;">
        <div style="font-size:.95rem;font-weight:800;color:#374151;letter-spacing:-.02em;margin-bottom:.3rem;">⚡ EnergyIQ</div>
        <div style="color:#9ca3af;font-size:.8rem;">Built with Streamlit · Random Forest trained on EIA RECS · Incentives: DSIRE · Address: OpenStreetMap Nominatim</div>
    </div>
    """, unsafe_allow_html=True)
