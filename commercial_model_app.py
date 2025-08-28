import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import altair as alt
import io
from datetime import datetime

# =================================
# Page Config & Styling
# =================================
st.set_page_config(
    page_title="B2B Fixed Line Commercial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* --- Main Layout & Font --- */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* --- Enhanced Custom Card Container --- */
    .input-card {
        /* Background and Border */
        background: #f8f9fa; /* A slightly off-white background for depth */
        border: 1px solid #e0e0e0; /* Softer border color */
        border-radius: 12px; /* Slightly more rounded corners */
        border-top: 4px solid #4a90e2; /* A bold, colored top border */
        position: relative;
        overflow: hidden;
        
        /* Box Shadow */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        
        /* Spacing and Transition */
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease-in-out;
    }

    /* --- Hover Effect for Interactivity --- */
    .input-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px -5px rgba(0, 0, 0, 0.15), 0 6px 8px -3px rgba(0, 0, 0, 0.08);
    }

    /* --- Blue Header for Input Cards --- */
    .card-header {
        background-color: #4f46e5;
        color: white;
        padding: 0.75rem 1.25rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        border-radius: 0.75rem 0.75rem 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* --- Input Widget Styling --- */
    label[data-baseweb="form-control-label"] { font-size: 0.9rem !important; font-weight: 500 !important; color: #4a5568 !important; }
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #f1f5f9; border: 1px solid #cbd5e0; border-radius: 0.375rem; height: 42px; font-size: 1.05rem;
    }
    .stForm [data-testid="stButton"] button {
        background-color: transparent; border: 2px solid #4a5568; color: #4a5568; font-weight: 600; padding: 0.6rem 1.2rem; transition: all 0.2s;
    }
    .stForm [data-testid="stButton"] button:hover { background-color: #4a5568; color: white; border-color: #4a5568; }

    /* --- Enhanced KPI Dashboard Styling --- */
    [data-testid="stMetric"] {
        /* Background and Border */
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 12px; /* Slightly more rounded corners */
        
        /* Box Shadow */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        
        /* Padding and Transition */
        padding: 1.5rem; /* Increased padding for more space */
        transition: all 0.3s ease-in-out;
    }

    /* --- Hover Effect --- */
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* --- KPI Value Styling --- */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important; /* Larger font size for prominence */
        font-weight: 700 !important;
        color: #1e293b;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }

    /* --- KPI Label Styling --- */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important; /* Slightly larger and more readable */
        color: #64748b;
        font-weight: 500; /* Medium weight for better readability */
        letter-spacing: 0.5px;
    }
    
    /* --- Payback Banner Styling --- */
    .payback-banner {
        background-color: #dcfce7; /* Green-100 */
        color: #166534; /* Green-800 */
        padding: 0.75rem 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: 500;
        font-size: 1rem;
        margin-top: 1rem;
    }

    /* --- Placeholder for Results --- */
    .placeholder { text-align: center; padding: 3rem; border: 2px dashed #e2e8f0; border-radius: 0.75rem; }
    /* --- Widget Styling (Labels) --- */
    label[data-baseweb="form-control-label"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #4a5568 !important; /* Gray-700 */
    }

    /* --- Widget Styling (Inputs & Selectboxes) --- */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        font-size: 1.05rem !important;
        height: 42px !important;
        padding-left: 0.75rem !important;
        border-radius: 0.375rem !important;
        border-color: #cbd5e0 !important; /* Gray-300 */
    }
    
    /* --- Specific Highlight for MRC Input --- */
    .highlight-input input {
        border-color: #e53e3e !important; /* Red-500 */
        box-shadow: 0 0 0 1px #e53e3e !important;
    }

    /* --- Button Styling --- */
    .stButton > button {
        border-radius: 0.375rem;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        font-size: 1.05rem;
    }
    .stForm [data-testid="stButton"] button {
        background-color: #e53e3e; /* Red-500 */
        color: white;
    }
    .stForm [data-testid="stButton"] button:hover {
        background-color: #c53030; /* Red-600 */
        color: white;
        border-color: #c53030;
    }

    /* --- Section Headers & Metrics --- */
    .section-header { background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%); color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.25rem; }
    [data-testid="metric-container"] { background-color: #e8f7f7; border: 1px solid #E0E0E0; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# =================================
# Helper Functions (Business Logic - Unchanged)
# =================================
CAPEX_NAMES = {"Dismantle", "License", "Unlicense", "Fiber", "Other Cost"}
# ... (All helper functions remain the same)
# ADD THIS NEW FUNCTION
def calculate_irr(values, guess=0.1, tol=1e-12, max_iter=100):
    """
    Calculate the Internal Rate of Return (IRR) using the Newton-Raphson method.
    This is a pure Python + NumPy replacement for numpy_financial.irr
    """
    values = np.atleast_1d(values)
    if values.size == 0:
        return np.nan

    rate = guess
    for _ in range(max_iter):
        # Calculate NPV and its derivative (dNPV/dr)
        t = np.arange(len(values))
        npv = np.sum(values / ((1 + rate) ** t))
        d_npv = np.sum(-t * values / ((1 + rate) ** (t + 1)))

        # Avoid division by zero
        if abs(d_npv) < 1e-12:
            return np.nan # Or handle as an error/special case

        # Newton-Raphson iteration
        new_rate = rate - npv / d_npv
        
        # Check for convergence
        if abs(new_rate - rate) < tol:
            return new_rate
        
        rate = new_rate
        
    return np.nan # Failed to converge
def fmt_currency(x: float, currency: str) -> str:
    try:
        if abs(x) >= 1_000_000: return f"{x/1_000_000:,.2f}M {currency}"
        elif abs(x) >= 1_000: return f"{x/1_000:,.2f}K {currency}"
        return f"{x:,.2f} {currency}"
    except (ValueError, TypeError): return f"- {currency}"
def fmt_pct(x: float) -> str:
    try: return f"{x:,.2f}%"
    except (ValueError, TypeError): return "- %"
@st.cache_data(ttl=300, show_spinner="Parsing workbook...")
def load_data_from_upload(uploaded_file) -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(uploaded_file)
        data = { "Product": pd.read_excel(xls, "Product"), "CostItems": pd.read_excel(xls, "CostItems"), "ProductCostRelation": pd.read_excel(xls, "ProductCostRelation"), "CostTable_USD": pd.read_excel(xls, "CostTable_USD"), "CostTable_MMK": pd.read_excel(xls, "CostTable_MMK")}
        for df in data.values():
            if "CostItemID" in df.columns: df["CostItemID"] = df["CostItemID"].astype(str)
            if "ProductID" in df.columns: df["ProductID"] = df["ProductID"].astype(str)
        return data
    except Exception as e:
        st.error(f"❌ File processing error: {e}"); return {}
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
        a, b = s.split('-',1);
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
def compute_costs_for_product(product_id: str, currency_table: pd.DataFrame, rel_df: pd.DataFrame, items_df: pd.DataFrame, inputs: Dict[str, float], choices: Dict[str, str]) -> Tuple[float, float, List[Dict]]:
    related = rel_df[rel_df["ProductID"] == product_id]
    breakdown, opex, capex = [], 0.0, 0.0
    for _, rel in related.iterrows():
        cid = rel["CostItemID"]
        item_row = items_df.loc[items_df["CostItemID"] == cid]
        if item_row.empty: continue
        item_name = item_row.iloc[0]["ItemName"]
        is_capex = False if cid == "I02" else item_name in CAPEX_NAMES
        if choices.get(cid, "").upper() == "NA":
            breakdown.append({"CostItemID": cid, "ItemName": item_name, "VariableUsed": "NA (Excluded)", "Amount": 0.0, "Type": "CAPEX" if is_capex else "OPEX"})
            continue
        rows = currency_table[currency_table["CostItemID"] == cid].copy()
        if rows.empty: continue
        chosen_var = choices.get(cid)
        if chosen_var:
            match = rows[rows["Variable"].astype(str) == str(chosen_var)]
            row = match.iloc[0] if not match.empty else rows.iloc[0]
        else: row = pick_variable_row(rows, inputs)
        amount = float(row["CostAmount"])
        if cid == "I03": amount *= inputs.get("Bandwidth", 1.0)
        breakdown.append({"CostItemID": cid, "ItemName": item_name, "VariableUsed": str(row.get("Variable", "")).strip() or "Auto", "Amount": amount, "Type": "CAPEX" if is_capex else "OPEX"})
        if is_capex: capex += amount
        else: opex += amount
    return opex, capex, breakdown
def discounted_series(values: List[float], rate: float) -> np.ndarray:
    m, monthly_rate = len(values), rate / 12.0
    months = np.arange(1, m + 1)
    df = 1.0 / (1.0 + monthly_rate) ** months
    return np.array(values) * df
def init_state():
    defaults = {'scenario_name': 'Base Scenario', 'currency': 'USD', 'months': 12, 'discount_rate': 0.25, 'bandwidth': 500.0, 'fiber_distance': 0.0, 'site_count': 1, 'mrc': 1250.0, 'otc': 0.0, 'product_id': None, 'manual_variable_choices': {}, 'calc_done': False, 'results': None, 'scenarios': {}}
    for k, v in defaults.items(): st.session_state.setdefault(k, v)

init_state()

# =================================
# Sidebar
# =================================
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded_file = st.file_uploader("Upload source.xlsx", type="xlsx")
    st.divider()
    st.markdown("### ⚙️ Scenarios")
    st.session_state.scenario_name = st.text_input("Current Scenario Name", st.session_state.scenario_name)
    c1, c2 = st.columns(2)
    if c1.button("💾 Save Scenario", use_container_width=True, disabled=not st.session_state.calc_done):
        key = f"{st.session_state.scenario_name} ({datetime.now().strftime('%H:%M:%S')})"
        st.session_state.scenarios[key] = st.session_state.results
        st.toast(f"Scenario '{key}' saved!", icon="💾")
    if c2.button("🗑️ Clear All", use_container_width=True):
        st.session_state.scenarios = {}; st.toast("All saved scenarios cleared.", icon="🗑️")
    if st.session_state.scenarios:
        with st.expander("Saved Scenarios", expanded=True): st.write(list(st.session_state.scenarios.keys()))

# =================================
# Main App Body
# =================================
st.title("B2B Fixed Line Commercial Model")

if not uploaded_file:
    st.info("👋 Welcome! Please upload your `source.xlsx` file using the sidebar to begin."); st.stop()

data = load_data_from_upload(uploaded_file)
is_valid, err_msg = validate_data(data)
if not is_valid:
    st.error(f"❌ **Invalid File:** {err_msg}"); st.stop()

# --- Main Two-Column Layout ---
left_col, right_col = st.columns([2, 3])

# --- INPUTS COLUMN (LEFT) ---
with left_col:
    # Card 1: Primary Selections
    # --- STEP 1: Primary Selections ---
    with st.container(border=True):
        st.markdown('<div class="section-header" style="margin-top:0.5rem; margin-bottom:1.5rem;">1. Primary Selections</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        product_df = data["Product"]
        if st.session_state.product_id is None and not product_df.empty:
            st.session_state.product_id = product_df["ProductID"].iloc[0]
        
        with c1:
            st.session_state.product_id = st.selectbox("Product Type", options=product_df["ProductID"], format_func=lambda pid: f"{pid} — {product_df.loc[product_df['ProductID']==pid, 'ProductName'].values[0]}", index=list(product_df["ProductID"]).index(st.session_state.product_id))
        with c2:
            st.session_state.currency = st.selectbox("Currency", ["USD", "MMK"], index=["USD", "MMK"].index(st.session_state.currency))

    st.write("") # Spacer

    # --- STEP 2: Configure Details ---
    with st.container(border=True):
        st.markdown('<div class="section-header" style="margin-top:0.5rem; margin-bottom:1.5rem;">2. Configure Scenario Details</div>', unsafe_allow_html=True)
        with st.form("model_inputs"):
            st.markdown("##### Revenue")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="highlight-input">', unsafe_allow_html=True)
                st.session_state.mrc = st.number_input("MRC (Monthly)", 0.0, value=st.session_state.mrc, step=50.0, format="%.2f")
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.session_state.otc = st.number_input("OTC (One-Time)", 0.0, value=st.session_state.otc, step=100.0, format="%.2f")

            st.markdown("##### Contract & Cost Drivers")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.months = st.number_input("Contract Months", 1, 120, st.session_state.months, 1)
                st.session_state.bandwidth = st.number_input("Bandwidth (Mbps)", 0.0, value=st.session_state.bandwidth, step=10.0)
            with c2:
                st.session_state.discount_rate = st.number_input("Annual Discount Rate (%)", 0.0, 100.0, st.session_state.discount_rate * 100, 1.0) / 100.0
                st.session_state.fiber_distance = st.number_input("Fiber Distance (km)", 0.0, value=st.session_state.fiber_distance, step=0.5)

            with st.expander("Advanced Cost Overrides (Optional)"):
                cost_table = data[f"CostTable_{st.session_state.currency}"]
                related_items = data["ProductCostRelation"][data["ProductCostRelation"]["ProductID"] == st.session_state.product_id]
                if not related_items.empty:
                    override_cols = st.columns(2)
                    for i, (_, relrow) in enumerate(related_items.iterrows()):
                        cid, rows = relrow["CostItemID"], cost_table[cost_table["CostItemID"] == relrow["CostItemID"]]
                        distinct_vars = ["NA"] + sorted([v for v in rows["Variable"].dropna().astype(str).str.strip().unique() if v])
                        if len(distinct_vars) > 1:
                            item_name = data["CostItems"].loc[data["CostItems"]["CostItemID"] == cid, "ItemName"].iloc[0]
                            current_choice, default_index = st.session_state.manual_variable_choices.get(cid), 0
                            if current_choice and current_choice in distinct_vars: default_index = distinct_vars.index(current_choice)
                            with override_cols[i % 2]:
                                choice = st.selectbox(f"{item_name} ({cid})", options=distinct_vars, index=default_index, key=f"var_override_{st.session_state.product_id}_{cid}")
                                st.session_state.manual_variable_choices[cid] = choice
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Calculate & Analyze", use_container_width=True)
            if submitted: st.session_state.calc_done = True
        st.markdown('</div>', unsafe_allow_html=True)

# --- Perform Calculation ---
if st.session_state.calc_done:
    with st.spinner("Updating analysis..."):
        user_inputs = {"Bandwidth": st.session_state.bandwidth, "Fiber Distance": st.session_state.fiber_distance, "No of Site": st.session_state.site_count}
        opex_m, capex_t, breakdown = compute_costs_for_product(st.session_state.product_id, data[f"CostTable_{st.session_state.currency}"], data["ProductCostRelation"], data["CostItems"], user_inputs, st.session_state.manual_variable_choices)
        m, mrc, otc = st.session_state.months, st.session_state.mrc, st.session_state.otc
        TR_m = [mrc] * m; TR_m[0] += otc
        reg_fee_m = [tr * 0.04 for tr in TR_m]
        otc_cost = next((i["Amount"] for i in breakdown if i["CostItemID"] == "I02"), 0.0)
        reg_opex_m = opex_m - otc_cost
        OPEX_m = [reg_opex_m] * m; OPEX_m[0] += otc_cost
        OPEX_m = [o + rf for o, rf in zip(OPEX_m, reg_fee_m)]
        EBITDA_m = [tr - oc for tr, oc in zip(TR_m, OPEX_m)]
        DEBITDA_m = discounted_series(EBITDA_m, st.session_state.discount_rate)
        NPV = float(np.sum(DEBITDA_m)) - capex_t
        cum_DEBITDA = np.cumsum(DEBITDA_m)
        TR_t, OPEX_t = sum(TR_m), sum(OPEX_m)
        EBITDA_t = TR_t - OPEX_t
        EBITDA_pct = (EBITDA_t / TR_t * 100.0) if TR_t > 0 else 0.0
        payback_idx = np.argmax(cum_DEBITDA - capex_t > 0) if np.any(cum_DEBITDA - capex_t > 0) else None
        payback = int(payback_idx + 1) if payback_idx is not None else None
        years = int(np.ceil(m / 12))
        DEBITDA_y = [float(np.sum(DEBITDA_m[y*12:min((y+1)*12, m)])) for y in range(years)]
        cf_y = [-capex_t] + DEBITDA_y
        irr = calculate_irr(cf_y) # Use our new custom function
        irr_pct = irr * 100 if irr is not None and not np.isnan(irr) else None
        series_df = pd.DataFrame({"Month": range(1, m + 1), "TR_monthly": TR_m, "OPEX_monthly": OPEX_m, "EBITDA_monthly": EBITDA_m, "DEBITDA_monthly": DEBITDA_m, "Cum_DEBITDA": cum_DEBITDA, "Net_Cum_Cash_Flow": cum_DEBITDA - capex_t})
        st.session_state.results = {'TR_total': TR_t, 'OPEX_total': OPEX_t, 'CAPEX_total': capex_t, 'EBITDA_total': EBITDA_t, 'EBITDA_pct': EBITDA_pct, 'NPV': NPV, 'IRR_annual_pct': irr_pct, 'discounted_payback': payback, 'breakdown': breakdown, 'series': series_df, 'context': {'currency': st.session_state.currency, 'name': st.session_state.scenario_name}}

# --- RESULTS COLUMN (RIGHT) ---
with right_col:
    if st.session_state.calc_done and 'results' in st.session_state and st.session_state.results:
        res, currency = st.session_state.results, st.session_state.results['context']['currency']
        st.subheader(f"Analysis for '{res['context']['name']}'")
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Total Revenue", fmt_currency(res['TR_total'], currency))
        kpi_cols[1].metric("Total OPEX", fmt_currency(res['OPEX_total'], currency))
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Total CAPEX", fmt_currency(res['CAPEX_total'], currency))
        kpi_cols[1].metric("Total EBITDA", fmt_currency(res['EBITDA_total'], currency))
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Net Cash Flow", fmt_currency(res['EBITDA_total'] - res['CAPEX_total'], currency))
        kpi_cols[1].metric("EBITDA Margin", fmt_pct(res['EBITDA_pct']))
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Net Present Value (NPV)", fmt_currency(res['NPV'], currency))
        kpi_cols[1].metric("Annual IRR", fmt_pct(res['IRR_annual_pct']) if res['IRR_annual_pct'] is not None else "N/A")
        
        if res['discounted_payback']:
            st.markdown(f'<div class="payback-banner">🕒 Discounted Payback Period: Achieved in {res["discounted_payback"]} months</div>', unsafe_allow_html=True)
        else:
            st.warning("🕒 Discounted Payback Period: Not reached within the contract term.")
    else:
        st.markdown('<div class="placeholder">📊<br>Your analysis results will appear here.</div>', unsafe_allow_html=True)

# --- DEEP DIVE SECTION (FULL WIDTH) ---
if st.session_state.calc_done and 'results' in st.session_state and st.session_state.results:
    res = st.session_state.results
    st.markdown("---")
    st.subheader("Deep Dive Analysis")
    
    tab_tables, tab_charts, tab_compare = st.tabs(["📋 Detailed Breakdowns", "📈 Charts", "⚖️ Compare Scenarios"])
    
    with tab_tables:
        st.subheader("Cost Item Breakdown")
        df_break = pd.DataFrame(res['breakdown'])
        if not df_break.empty:
            st.dataframe(df_break.sort_values(["Type", "CostItemID"]), use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format=f"{res['context']['currency']} %.2f")})
            csv_breakdown = df_break.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Cost Breakdown", data=csv_breakdown, file_name=f"cost_breakdown.csv")
        else: st.info("No cost items were found for this product.")
        
        st.markdown("---")
        st.subheader("Monthly Diagnostic Series")
        diag_df = res['series']
        st.dataframe(diag_df, use_container_width=True, hide_index=True, column_config={col: st.column_config.NumberColumn(format="%.2f") for col in diag_df.columns if col != 'Month'})
        csv_diag = diag_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Monthly Series", data=csv_diag, file_name=f"diagnostic_series.csv")

    with tab_charts:
        st.subheader("Monthly EBITDA (TR - OPEX)")
        chart_df = res['series']
        cashflow_chart = alt.Chart(chart_df).mark_bar().encode(x=alt.X("Month:O", title="Contract Month"), y=alt.Y('EBITDA_monthly:Q', title=f'Amount ({res["context"]["currency"]})'), color=alt.condition(alt.datum.EBITDA_monthly > 0, alt.value("#28a745"), alt.value("#dc3545")), tooltip=['Month', alt.Tooltip('TR_monthly:Q', format=',.2f'), alt.Tooltip('OPEX_monthly:Q', format=',.2f'), alt.Tooltip('EBITDA_monthly:Q', format=',.2f')]).interactive()
        st.altair_chart(cashflow_chart, use_container_width=True)
        st.subheader("Cumulative Discounted Payback")
        base = alt.Chart(chart_df).encode(x=alt.X("Month:O", title="Contract Month"))
        payback_chart = base.mark_area(line={'color':'#6f42c1'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#6f42c1', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(y=alt.Y('Net_Cum_Cash_Flow:Q', title=f'Cumulative Value ({res["context"]["currency"]})'), tooltip=['Month', alt.Tooltip('Net_Cum_Cash_Flow:Q', format=',.2f')])
        rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[3,3]).encode(y='y')
        st.altair_chart((payback_chart + rule).interactive(), use_container_width=True)

    with tab_compare:
        if not st.session_state.scenarios: st.info("No scenarios saved. Run a calculation and click '💾 Save Scenario' to compare.")
        else:
            st.subheader("Scenario Comparison")
            compare_data = [{"Scenario": name, "NPV": sc_res['NPV'], "IRR (%)": sc_res['IRR_annual_pct'], "EBITDA Margin (%)": sc_res['EBITDA_pct'], "Payback (Months)": sc_res['discounted_payback'] or "N/A", "Total Revenue": sc_res['TR_total'], "Total OPEX": sc_res['OPEX_total'], "Total CAPEX": sc_res['CAPEX_total']} for name, sc_res in st.session_state.scenarios.items()]
            df_compare = pd.DataFrame(compare_data)
            st.dataframe(df_compare, use_container_width=True, hide_index=True, column_config={"NPV": st.column_config.NumberColumn(format="%.2f"), "Total Revenue": st.column_config.NumberColumn(format="%.2f"), "Total OPEX": st.column_config.NumberColumn(format="%.2f"), "Total CAPEX": st.column_config.NumberColumn(format="%.2f")})
