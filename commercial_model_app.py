import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime

# =================================
# Page Config & Styling
# =================================
st.set_page_config(
    page_title="B2B Commercial Intelligence Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern CSS with enhanced UI/UX
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-color: #6366F1;
        --primary-dark: #4F46E5;
        --secondary-color: #8B5CF6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --dark-color: #1E293B;
        --light-bg: #F8FAFC;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
        --text-primary: #1E293B;
        --text-secondary: #64748B;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(180deg, #F0F4F8 0%, #FFFFFF 100%);
        padding-top: 0;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.1); opacity: 0.5; }
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Modern Card Container */
    .modern-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
    }
    
    .modern-card-header {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        padding: 1rem 1.5rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        border-radius: 16px 16px 0 0;
        font-weight: 600;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .modern-card-header-icon {
        width: 24px;
        height: 24px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Enhanced Input Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: var(--light-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        outline: none;
    }
    
    /* Label Styling */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Form Submit Button */
    .stForm [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--success-color), #059669);
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
    }
    
    /* Enhanced Metrics */
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Success Banner */
    .success-banner {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border: 1px solid #10B981;
        color: #065F46;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Warning Banner */
    .warning-banner {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 1px solid var(--warning-color);
        color: #92400E;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-bg);
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--light-bg);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--light-bg);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        transition: all 0.2s;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--card-bg);
        box-shadow: var(--shadow-sm);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: var(--card-bg);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: var(--light-bg);
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background: var(--card-bg);
        border-color: var(--primary-dark);
        box-shadow: var(--shadow-sm);
    }
    
    /* DataFrames */
    .dataframe {
        font-size: 0.9rem;
        border: none !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
        color: white !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe tbody tr:hover {
        background: var(--light-bg) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# =================================
# Helper Functions (Business Logic)
# =================================
CAPEX_NAMES = {"Dismantle", "License", "Unlicense", "Fiber", "Other Cost"}

def fmt_currency(x: float, currency: str) -> str:
    try:
        if abs(x) >= 1_000_000: return f"{x/1_000_000:,.2f}M {currency}"
        elif abs(x) >= 1_000: return f"{x/1_000:,.2f}K {currency}"
        return f"{x:,.2f} {currency}"
    except (ValueError, TypeError): return f"- {currency}"

def fmt_pct(x: float) -> str:
    try: return f"{x:,.2f}%"
    except (ValueError, TypeError): return "- %"

import io

@st.cache_data(ttl=300, show_spinner="Loading data...")
def load_data_from_upload(uploaded_file) -> Dict[str, pd.DataFrame]:
    if uploaded_file is None:
        return {}

    try:
        # Reset buffer pointer (important!)
        uploaded_file.seek(0)

        # Wrap into BytesIO for xlrd/openpyxl compatibility
        file_bytes = io.BytesIO(uploaded_file.read())

        xls = pd.ExcelFile(file_bytes)

        required_sheets = ["Product", "CostItems", "ProductCostRelation", "CostTable_USD", "CostTable_MMK"]
        data = {sheet: pd.read_excel(xls, sheet) for sheet in required_sheets if sheet in xls.sheet_names}

        # Normalize IDs
        for df in data.values():
            if "CostItemID" in df.columns:
                df["CostItemID"] = df["CostItemID"].astype(str)
            if "ProductID" in df.columns:
                df["ProductID"] = df["ProductID"].astype(str)

        return data

    except Exception as e:
        st.error(f"❌ File processing error: {e}")
        st.exception(e)
        return {}

def validate_data(data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
    if not data: return False, "Data could not be loaded."
    required = ["Product", "CostItems", "ProductCostRelation", "CostTable_USD", "CostTable_MMK"]
    missing = [s for s in required if s not in data or data[s].empty]
    if missing: return False, f"Missing or empty sheets: {', '.join(missing)}"
    return True, ""

def parse_numeric_condition(expr: str, x: float) -> bool:
    if expr is None: return False
    s = str(expr).strip().replace('≤','<=').replace('≥','>=').replace('==', '=')
    if not s: return False
    if '-' in s and all(p.strip().replace('.','',1).isdigit() for p in s.split('-',1)):
        a, b = s.split('-',1)
        try: return float(a.strip()) <= x <= float(b.strip())
        except: pass
    for op in ['>=','<=','>','<','=']:
        if s.startswith(op):
            try:
                num = float(''.join(c for c in s[len(op):] if c.isdigit() or c=='.') or '0')
                if op == '>': return x > num
                if op == '>=': return x >= num
                if op == '<': return x < num
                if op == '<=': return x <= num
                if op == '=': return abs(x - num) < 1e-9
            except: return False
    try: return abs(x - float(''.join(c for c in s if c.isdigit() or c=='.'))) < 1e-9
    except: return False

def pick_variable_row(cost_rows: pd.DataFrame, user_inputs: Dict[str, float]) -> Optional[pd.Series]:
    if cost_rows.empty: return None
    na_rows = cost_rows[cost_rows["Variable"].astype(str).str.strip().str.upper() == "NA"]
    if not na_rows.empty: return na_rows.iloc[0]
    if cost_rows["Variable"].fillna("").str.strip().eq("").all(): return cost_rows.iloc[0]
    for _, r in cost_rows.iterrows():
        cond = str(r.get("Variable", "")).strip()
        if not cond: continue
        for val in user_inputs.values():
            if parse_numeric_condition(cond, float(val)): return r
    return cost_rows.iloc[0]

def compute_costs_for_product(product_id: str, currency_table: pd.DataFrame, rel_df: pd.DataFrame, 
                             items_df: pd.DataFrame, inputs: Dict[str, float], 
                             choices: Dict[str, str]) -> Tuple[float, float, List[Dict]]:
    related = rel_df[rel_df["ProductID"] == product_id]
    breakdown, opex, capex = [], 0.0, 0.0
    for _, rel in related.iterrows():
        cid = rel["CostItemID"]
        item_row = items_df.loc[items_df["CostItemID"] == cid]
        if item_row.empty: continue
        item_name = item_row.iloc[0]["ItemName"]
        is_capex = False if cid == "I02" else item_name in CAPEX_NAMES
        if choices.get(cid, "").upper() == "NA":
            breakdown.append({
                "CostItemID": cid, "ItemName": item_name, "VariableUsed": "NA (Excluded)", 
                "Amount": 0.0, "Type": "CAPEX" if is_capex else "OPEX"
            })
            continue
        rows = currency_table[currency_table["CostItemID"] == cid].copy()
        if rows.empty: continue
        chosen_var = choices.get(cid)
        if chosen_var:
            match = rows[rows["Variable"].astype(str) == str(chosen_var)]
            row = match.iloc[0] if not match.empty else rows.iloc[0]
        else: 
            row = pick_variable_row(rows, inputs)
        amount = float(row["CostAmount"])
        if cid == "I03": amount *= inputs.get("Bandwidth", 1.0)
        breakdown.append({
            "CostItemID": cid, "ItemName": item_name, 
            "VariableUsed": str(row.get("Variable", "")).strip() or "Auto", 
            "Amount": amount, "Type": "CAPEX" if is_capex else "OPEX"
        })
        if is_capex: capex += amount
        else: opex += amount
    return opex, capex, breakdown

def discounted_series(values: List[float], rate: float) -> np.ndarray:
    m, monthly_rate = len(values), rate / 12.0
    months = np.arange(1, m + 1)
    df = 1.0 / (1.0 + monthly_rate) ** months
    return np.array(values) * df

def init_state():
    defaults = {
        'scenario_name': 'Base Scenario', 
        'currency': 'USD', 
        'months': 12, 
        'discount_rate': 0.25, 
        'bandwidth': 500.0, 
        'fiber_distance': 0.0, 
        'site_count': 1, 
        'mrc': 1250.0, 
        'otc': 0.0, 
        'product_id': None, 
        'manual_variable_choices': {}, 
        'calc_done': False, 
        'results': None, 
        'scenarios': {}
    }
    for k, v in defaults.items(): 
        st.session_state.setdefault(k, v)

init_state()

# =================================
# Sidebar
# =================================
with st.sidebar:
    with st.container():
        st.markdown("## 📊 Control Panel")
        st.markdown("---")
        
        st.markdown("### 📁 Data Source")
        uploaded_file = st.file_uploader("Upload source.xlsx", type="xlsx")

        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
        else:
            st.info("📤 Please upload a file")
        
        st.markdown("---")
        st.markdown("### 💾 Scenario Management")
        
        st.session_state.scenario_name = st.text_input(
            "Scenario Name", 
            st.session_state.scenario_name,
            help="Give your scenario a descriptive name"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True, disabled=not st.session_state.calc_done):
                key = f"{st.session_state.scenario_name} ({datetime.now().strftime('%H:%M')})"
                st.session_state.scenarios[key] = st.session_state.results
                st.success(f"Saved: {key}")
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.scenarios = {}
                st.info("All scenarios cleared")
        
        if st.session_state.scenarios:
            st.markdown("---")
            st.markdown("### 📋 Saved Scenarios")
            for idx, scenario_name in enumerate(st.session_state.scenarios.keys()):
                st.markdown(f"• {scenario_name}")

# =================================
# Main App
# =================================
# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">💼 B2B Commercial Intelligence Platform</h1>
    <p class="hero-subtitle">Advanced Fixed Line Revenue & Cost Modeling with NPV Analysis</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    # Welcome Screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: white; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <img src='https://cdn-icons-png.flaticon.com/512/3135/3135755.png' width='120' style='margin-bottom: 1rem;'>
        <h2 style='color: #1E293B; margin-bottom: 1rem;'>Welcome to Commercial Intelligence</h2>
        <p style='color: #64748B; font-size: 1.1rem;'>
        Upload your source data file to begin analyzing B2B fixed line commercial models.
        </p>
        <p style='color: #94A3B8; margin-top: 2rem;'>
        👈 Use the sidebar to upload your <code>source.xlsx</code> file
        </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# Load and validate data
data = load_data_from_upload(uploaded_file)
is_valid, err_msg = validate_data(data)
if not is_valid:
    st.error(f"❌ **Invalid File:** {err_msg}")
    st.stop()

# Main Layout
left_col, right_col = st.columns([5, 7])

# Left Column - Inputs
with left_col:
    # Product Selection Card
    st.markdown("""
    <div class="modern-card">
        <div class="modern-card-header">
            <div class="modern-card-header-icon">🎯</div>
            Step 1: Product Configuration
        </div>
    </div>
    """, unsafe_allow_html=True, help=False)
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        product_df = data["Product"]
        if st.session_state.product_id is None and not product_df.empty:
            st.session_state.product_id = product_df["ProductID"].iloc[0]
        
        with col1:
            st.session_state.product_id = st.selectbox(
                "📦 Product Type",
                options=product_df["ProductID"],
                format_func=lambda pid: f"{pid} — {product_df.loc[product_df['ProductID']==pid, 'ProductName'].values[0]}",
                index=list(product_df["ProductID"]).index(st.session_state.product_id)
            )
        
        with col2:
            st.session_state.currency = st.selectbox(
                "💱 Currency",
                ["USD", "MMK"],
                index=["USD", "MMK"].index(st.session_state.currency)
            )
    
    # Scenario Details Card
    st.markdown("""
    <div class="modern-card">
        <div class="modern-card-header">
            <div class="modern-card-header-icon">⚙️</div>
            Step 2: Commercial Parameters
        </div>
    </div>
    """, unsafe_allow_html=True, help=False)
    
    with st.form("model_inputs"):
        # Revenue Section
        st.markdown("#### 💰 Revenue Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.mrc = st.number_input(
                "Monthly Recurring Charge",
                0.0,
                value=st.session_state.mrc,
                step=50.0,
                format="%.2f",
                help="Monthly subscription fee"
            )
        with col2:
            st.session_state.otc = st.number_input(
                "One-Time Charge",
                0.0,
                value=st.session_state.otc,
                step=100.0,
                format="%.2f",
                help="Initial setup fee"
            )
        
        st.markdown("---")
        
        # Contract Details
        st.markdown("#### 📋 Contract & Technical Details")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.months = st.number_input(
                "Contract Duration (Months)",
                1, 120,
                st.session_state.months,
                1,
                help="Total contract length"
            )
            st.session_state.bandwidth = st.number_input(
                "Bandwidth (Mbps)",
                0.0,
                value=st.session_state.bandwidth,
                step=10.0,
                help="Service bandwidth requirement"
            )
        with col2:
            st.session_state.discount_rate = st.number_input(
                "Annual Discount Rate (%)",
                0.0, 100.0,
                st.session_state.discount_rate * 100,
                1.0,
                help="For NPV calculations"
            ) / 100.0
            st.session_state.fiber_distance = st.number_input(
                "Fiber Distance (km)",
                0.0,
                value=st.session_state.fiber_distance,
                step=0.5,
                help="Total fiber cable distance"
            )
        
        # Advanced Options
        with st.expander("🔧 Advanced Cost Overrides"):
            cost_table = data[f"CostTable_{st.session_state.currency}"]
            related_items = data["ProductCostRelation"][
                data["ProductCostRelation"]["ProductID"] == st.session_state.product_id
            ]
            
            if not related_items.empty:
                for _, relrow in related_items.iterrows():
                    cid = relrow["CostItemID"]
                    rows = cost_table[cost_table["CostItemID"] == cid]
                    distinct_vars = ["NA"] + sorted([
                        v for v in rows["Variable"].dropna().astype(str).str.strip().unique() if v
                    ])
                    
                    if len(distinct_vars) > 1:
                        item_name = data["CostItems"].loc[
                            data["CostItems"]["CostItemID"] == cid, "ItemName"
                        ].iloc[0]
                        
                        current_choice = st.session_state.manual_variable_choices.get(cid)
                        default_index = 0
                        if current_choice and current_choice in distinct_vars:
                            default_index = distinct_vars.index(current_choice)
                        
                        choice = st.selectbox(
                            f"{item_name}",
                            options=distinct_vars,
                            index=default_index,
                            key=f"var_{cid}"
                        )
                        st.session_state.manual_variable_choices[cid] = choice
        
        # Submit Button
        submitted = st.form_submit_button(
            "🚀 Calculate NPV & Analysis",
            use_container_width=True
        )
        if submitted:
            st.session_state.calc_done = True

# Perform Calculations
if st.session_state.calc_done:
    with st.spinner("🔄 Running financial analysis..."):
        # Calculation logic remains the same
        user_inputs = {
            "Bandwidth": st.session_state.bandwidth,
            "Fiber Distance": st.session_state.fiber_distance,
            "No of Site": st.session_state.site_count
        }
        
        opex_m, capex_t, breakdown = compute_costs_for_product(
            st.session_state.product_id,
            data[f"CostTable_{st.session_state.currency}"],
            data["ProductCostRelation"],
            data["CostItems"],
            user_inputs,
            st.session_state.manual_variable_choices
        )
        
        m = st.session_state.months
        mrc = st.session_state.mrc
        otc = st.session_state.otc
        
        TR_m = [mrc] * m
        TR_m[0] += otc
        reg_fee_m = [tr * 0.04 for tr in TR_m]
        otc_cost = next((i["Amount"] for i in breakdown if i["CostItemID"] == "I02"), 0.0)
        reg_opex_m = opex_m - otc_cost
        OPEX_m = [reg_opex_m] * m
        OPEX_m[0] += otc_cost
        OPEX_m = [o + rf for o, rf in zip(OPEX_m, reg_fee_m)]
        EBITDA_m = [tr - oc for tr, oc in zip(TR_m, OPEX_m)]
        DEBITDA_m = discounted_series(EBITDA_m, st.session_state.discount_rate)
        NPV = float(np.sum(DEBITDA_m)) - capex_t
        cum_DEBITDA = np.cumsum(DEBITDA_m)
        TR_t = sum(TR_m)
        OPEX_t = sum(OPEX_m)
        EBITDA_t = TR_t - OPEX_t
        EBITDA_pct = (EBITDA_t / TR_t * 100.0) if TR_t > 0 else 0.0
        payback_idx = np.argmax(cum_DEBITDA - capex_t > 0) if np.any(cum_DEBITDA - capex_t > 0) else None
        payback = int(payback_idx + 1) if payback_idx is not None else None
        years = int(np.ceil(m / 12))
        DEBITDA_y = [float(np.sum(DEBITDA_m[y*12:min((y+1)*12, m)])) for y in range(years)]
        cf_y = [-capex_t] + DEBITDA_y
        irr = npf.irr(cf_y)
        irr_pct = irr * 100 if irr is not None and not np.isnan(irr) else None
        
        series_df = pd.DataFrame({
            "Month": range(1, m + 1),
            "TR_monthly": TR_m,
            "OPEX_monthly": OPEX_m,
            "EBITDA_monthly": EBITDA_m,
            "DEBITDA_monthly": DEBITDA_m,
            "Cum_DEBITDA": cum_DEBITDA,
            "Net_Cum_Cash_Flow": cum_DEBITDA - capex_t
        })
        
        st.session_state.results = {
            'TR_total': TR_t,
            'OPEX_total': OPEX_t,
            'CAPEX_total': capex_t,
            'EBITDA_total': EBITDA_t,
            'EBITDA_pct': EBITDA_pct,
            'NPV': NPV,
            'IRR_annual_pct': irr_pct,
            'discounted_payback': payback,
            'breakdown': breakdown,
            'series': series_df,
            'context': {
                'currency': st.session_state.currency,
                'name': st.session_state.scenario_name
            }
        }

# Right Column - Results
with right_col:
    if st.session_state.calc_done and st.session_state.results:
        res = st.session_state.results
        currency = res['context']['currency']
        
        # Results Header
        st.markdown(f"### 📊 Analysis Results: *{res['context']['name']}*")
        
        # KPI Dashboard
        st.markdown("#### Key Performance Indicators")
        
        # First Row of KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Total Revenue", fmt_currency(res['TR_total'], currency))
        with col2:
            st.metric("📉 Total OPEX", fmt_currency(res['OPEX_total'], currency))
        with col3:
            st.metric("🏗️ Total CAPEX", fmt_currency(res['CAPEX_total'], currency))
        
        # Second Row of KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 EBITDA", fmt_currency(res['EBITDA_total'], currency))
        with col2:
            st.metric("💹 EBITDA Margin", fmt_pct(res['EBITDA_pct']))
        with col3:
            st.metric("💵 Net Cash Flow", fmt_currency(res['EBITDA_total'] - res['CAPEX_total'], currency))
        
        # Third Row - Financial Metrics
        col1, col2 = st.columns(2)
        with col1:
            npv_delta = "positive" if res['NPV'] > 0 else "negative"
            st.metric(
                "🎯 Net Present Value",
                fmt_currency(res['NPV'], currency),
                delta=npv_delta
            )
        with col2:
            irr_value = fmt_pct(res['IRR_annual_pct']) if res['IRR_annual_pct'] else "N/A"
            st.metric("📊 Annual IRR", irr_value)
        
        # Payback Period Banner
        if res['discounted_payback']:
            st.markdown(f"""
            <div class="success-banner">
                ✅ Discounted Payback achieved in <strong>{res['discounted_payback']} months</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-banner">
                ⚠️ Payback not achieved within contract period
            </div>
            """, unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
        <div style='text-align: center; padding: 4rem; background: white; border-radius: 16px; border: 2px dashed #E2E8F0;'>
            <h3 style='color: #64748B;'>📊 Analysis Results</h3>
            <p style='color: #94A3B8;'>Configure parameters and click Calculate to see results</p>
        </div>
        """, unsafe_allow_html=True)

# Deep Dive Section (Full Width)
if st.session_state.calc_done and st.session_state.results:
    st.markdown("---")
    st.markdown("## 🔍 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📋 Breakdowns", "📈 Visualizations", "⚖️ Scenario Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cost Breakdown")
            df_break = pd.DataFrame(res['breakdown'])
            if not df_break.empty:
                st.dataframe(
                    df_break.sort_values(["Type", "Amount"], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True
                )
        
        with col2:
            st.markdown("#### Monthly Cash Flow")
            st.dataframe(
                res['series'][['Month', 'TR_monthly', 'OPEX_monthly', 'EBITDA_monthly']].head(12),
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        # Create interactive Plotly charts
        col1, col2 = st.columns(2)
        
        with col1:
            # EBITDA Chart
            fig_ebitda = go.Figure()
            fig_ebitda.add_trace(go.Bar(
                x=res['series']['Month'],
                y=res['series']['EBITDA_monthly'],
                name='EBITDA',
                marker_color=np.where(res['series']['EBITDA_monthly'] > 0, '#10B981', '#EF4444')
            ))
            fig_ebitda.update_layout(
                title="Monthly EBITDA",
                xaxis_title="Month",
                yaxis_title=f"Amount ({currency})",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_ebitda, use_container_width=True)
        
        with col2:
            # Cumulative Cash Flow Chart
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Scatter(
                x=res['series']['Month'],
                y=res['series']['Net_Cum_Cash_Flow'],
                mode='lines+markers',
                name='Cumulative Cash Flow',
                line=dict(color='#6366F1', width=3),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
            fig_cf.add_hline(y=0, line_dash="dash", line_color="red")
            fig_cf.update_layout(
                title="Cumulative Discounted Cash Flow",
                xaxis_title="Month",
                yaxis_title=f"Amount ({currency})",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_cf, use_container_width=True)
    
    with tab3:
        if st.session_state.scenarios:
            compare_data = []
            for name, sc_res in st.session_state.scenarios.items():
                compare_data.append({
                    "Scenario": name,
                    "NPV": sc_res['NPV'],
                    "IRR (%)": sc_res['IRR_annual_pct'],
                    "EBITDA Margin (%)": sc_res['EBITDA_pct'],
                    "Payback (Months)": sc_res['discounted_payback'] or "N/A"
                })
            
            df_compare = pd.DataFrame(compare_data)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
            
            # Comparison Chart
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=df_compare['Scenario'],
                y=df_compare['NPV'],
                name='NPV',
                marker_color='#6366F1'
            ))
            fig_compare.update_layout(
                title="NPV Comparison Across Scenarios",
                xaxis_title="Scenario",
                yaxis_title=f"NPV ({currency})",
                height=350
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("💡 No saved scenarios yet. Save your current analysis to start comparing.")
