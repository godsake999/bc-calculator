import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import numpy_financial as npf

st.set_page_config(page_title="B2B Fixed Line Commercial Model", layout="wide")

# ------------------------------
# Load data from source.xlsx
# ------------------------------
@st.cache_data(ttl=60)
def load_data() -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile("source.xlsx")
        data = {
            "Product": pd.read_excel(xls, "Product"),
            "CostItems": pd.read_excel(xls, "CostItems"),
            "ProductCostRelation": pd.read_excel(xls, "ProductCostRelation"),
            "CostTable_USD": pd.read_excel(xls, "CostTable_USD"),
            "CostTable_MMK": pd.read_excel(xls, "CostTable_MMK"),
        }
        
        # Clean data - ensure CostItemID and ProductID are strings
        for df in data.values():
            if "CostItemID" in df.columns:
                df["CostItemID"] = df["CostItemID"].astype(str)
            if "ProductID" in df.columns:
                df["ProductID"] = df["ProductID"].astype(str)
                
        return data
    except Exception as e:
        st.error(f"Failed to load data from source.xlsx: {e}")
        # Return empty dataframes if file loading fails
        return {
            "Product": pd.DataFrame(columns=["ProductID", "ProductName", "Description"]),
            "CostItems": pd.DataFrame(columns=["CostItemID", "ItemName"]),
            "ProductCostRelation": pd.DataFrame(columns=["ProductID", "CostItemID"]),
            "CostTable_USD": pd.DataFrame(columns=["CostID", "CostItemID", "Variable", "CostAmount"]),
            "CostTable_MMK": pd.DataFrame(columns=["CostID", "CostItemID", "Variable", "CostAmount"]),
        }

data = load_data()

st.sidebar.write("I07 Variables:", data["CostTable_USD"][data["CostTable_USD"]["CostItemID"] == "I07"]["Variable"].unique())
st.sidebar.write("I08 Variables:", data["CostTable_USD"][data["CostTable_USD"]["CostItemID"] == "I08"]["Variable"].unique())

# ------------------------------
# Small helpers
# ------------------------------
CAPEX_NAMES = {"Dismantle", "License", "Unlicense", "Fiber", "Other Cost"}

def parse_numeric_condition(expr: str, x: float) -> bool:
    """Evaluate a simple numeric condition string against x.
    Supports: >, >=, <, <=, =, ==, between 'a-b', and unicode ≤ ≥
    Any trailing text is ignored (e.g., '1 site' -> 1).
    """
    if expr is None:
        return False
    s = str(expr).strip().replace('≤','<=').replace('≥','>=').replace('==', '=')
    if not s:
        return False
    # Handle range a-b
    if '-' in s and all(p.strip().replace('.','',1).isdigit() for p in s.split('-',1)):
        a, b = s.split('-',1)
        try:
            a, b = float(a.strip()), float(b.strip())
            return a <= x <= b
        except:
            pass
    # Extract leading comparator if present
    for op in ['>=','<=','>','<','=']:
        if s.startswith(op):
            try:
                num = float(''.join(ch for ch in s[len(op):] if (ch.isdigit() or ch=='.')) or '0')
                if op == '>': return x > num
                if op == '>=': return x >= num
                if op == '<': return x < num
                if op == '<=': return x <= num
                if op == '=': return abs(x - num) < 1e-9
            except:
                return False
    # fallback: pure number in string
    try:
        num = float(''.join(ch for ch in s if (ch.isdigit() or ch=='.')))
        return abs(x - num) < 1e-9
    except:
        return False

def pick_variable_row(cost_rows: pd.DataFrame, user_inputs: Dict[str, float]) -> Optional[pd.Series]:
    """Choose row by evaluating the Variable condition against relevant user inputs."""
    if cost_rows.empty:
        return None
    
    # Check for "NA" option first
    na_rows = cost_rows[cost_rows["Variable"].astype(str).str.strip().str.upper() == "NA"]
    if not na_rows.empty:
        return na_rows.iloc[0]
    
    # If no variable defined (all blank), just take first
    if cost_rows["Variable"].fillna("").str.strip().eq("").all():
        return cost_rows.iloc[0]

    # Try to find first matching condition
    for _, r in cost_rows.iterrows():
        cond = str(r.get("Variable", "")).strip()
        if not cond:
            continue
            
        # Extract all numbers from condition
        numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", cond)]
        if not numbers:
            continue
            
        # Check against all numeric inputs
        for input_val in user_inputs.values():
            if parse_numeric_condition(cond, float(input_val)):
                return r
                
    # Fallback to top row if no match
    return cost_rows.iloc[0]

def compute_costs_for_product(product_id: str,
                            currency_table: pd.DataFrame,
                            ProductCostRelation: pd.DataFrame,
                            CostItems: pd.DataFrame,
                            user_inputs: Dict[str, float],
                            manual_variable_choices: Dict[str, str]) -> Tuple[float, float, List[Dict]]:
    """Returns (OPEX_monthly, CAPEX_total, breakdown_list)"""
    related = ProductCostRelation[ProductCostRelation["ProductID"] == product_id]
    breakdown = []
    opex = 0.0
    capex = 0.0
    
    for _, rel in related.iterrows():
        cid = rel["CostItemID"]
        item = CostItems.loc[CostItems["CostItemID"] == cid].iloc[0]
        item_name = item["ItemName"]
        # Special handling for OTC (I02)
        if cid == "I02":
            is_otc = True
            is_capex = False  # OTC is typically treated as OPEX
        else:
            is_otc = False
            is_capex = item_name in CAPEX_NAMES

        # Check if user selected "NA" for this item
        if manual_variable_choices.get(cid, "").upper() == "NA":
            breakdown.append({
                "CostItemID": cid,
                "ItemName": item_name,
                "VariableUsed": "NA",
                "Amount": 0.0,
                "Type": "CAPEX" if is_capex else "OPEX"  # Maintain original type
            })
            continue  # Skip rest of processing for this item

        rows = currency_table[currency_table["CostItemID"] == cid].copy()
        if rows.empty:
            continue

        # Get the selected variable or auto-pick one
        chosen_var = manual_variable_choices.get(cid)
        if chosen_var:
            match = rows[rows["Variable"].astype(str) == str(chosen_var)]
            if not match.empty:
                row = match.iloc[0]
            else:
                row = rows.iloc[0]
        else:
            row = pick_variable_row(rows, user_inputs)

        amount = float(row["CostAmount"])
        var_used = str(row.get("Variable", "")).strip()

        # Special handling for Bandwidth cost (I03)
        if cid == "I03":
            amount *= user_inputs.get("Bandwidth", 1.0)

        breakdown.append({
            "CostItemID": cid,
            "ItemName": item_name,
            "VariableUsed": var_used,
            "Amount": amount,
            "Type": "CAPEX" if is_capex else "OPEX"
        })

        if is_capex:
            capex += amount
        else:
            opex += amount

    return opex, capex, breakdown

def discounted_series(values: List[float], annual_discount_rate: float) -> np.ndarray:
    """Discount a monthly series at a given annual rate."""
    m = len(values)
    monthly_rate = annual_discount_rate / 12.0
    months = np.arange(1, m + 1, dtype=float)
    df = 1.0 / (1.0 + monthly_rate) ** months
    return np.array(values, dtype=float) * df

def irr_newton(cashflows: List[float], guess: float = 0.1, tol: float = 1e-6, maxiter: int = 100) -> Optional[float]:
    """Simple IRR via Newton's method on monthly cashflows. Returns rate per period, or None if fails."""
    c = np.array(cashflows, dtype=float)
    r = guess
    for _ in range(maxiter):
        # NPV and derivative
        denom = (1 + r) ** np.arange(len(c))
        npv = np.sum(c / denom)
        dnpv = np.sum(-np.arange(len(c)) * c / ((1 + r) ** (np.arange(len(c)) + 1)))
        if abs(dnpv) < 1e-12:
            break
        new_r = r - npv / dnpv
        if abs(new_r - r) < tol:
            return new_r
        r = new_r
    return None

# ------------------------------
# UI
# ------------------------------
st.title("📈 B2B Fixed Line Commercial Model (Streamlit)")

with st.sidebar:
    st.subheader("Data Sources")
    st.caption("Using data from source.xlsx. Make sure the file is in the same directory.")
    st.markdown("---")
    st.caption("Tip: Use the Variable column in CostItems to define dependencies (e.g., 'Bandwidth>100')")
    if st.button("🔄 Reload Data from Excel"):
        st.cache_data.clear()
        data = load_data()
        st.rerun()

# Main inputs
left, right = st.columns([1,1])

with left:
    st.subheader("Inputs")
    product_df = data["Product"]
    if product_df.empty:
        st.error("No product data available. Please check source.xlsx")
        st.stop()
        
    product_label = st.selectbox(
        "Product Type",
        options=product_df["ProductID"],
        format_func=lambda pid: f"{pid} — {product_df.loc[product_df['ProductID']==pid, 'ProductName'].values[0]}",
        index=0
    )

    currency = st.selectbox("Currency", options=["USD", "MMK"], index=0)
    months = st.number_input("Contract Month", min_value=1, value=12, step=1)
    bandwidth = st.number_input("Bandwidth (Mbps)", min_value=0.0, value=100.0, step=1.0)
    fiber_distance = st.number_input("Fiber Distance (km)", min_value=0.0, value=2.0, step=0.5)
    site_count = st.number_input("No of Site", min_value=1, value=1, step=1)
    mrc = st.number_input("MRC (per month)", min_value=0.0, value=500.0, step=10.0)
    otc = st.number_input("OTC (one-time)", min_value=0.0, value=1000.0, step=10.0)
    discount_rate = st.number_input("Annual Discount Rate (for DEBITDA/NPV/DPB)", min_value=0.0, value=0.25, step=0.01, format="%.2f")

    user_inputs = {
        "Bandwidth": float(bandwidth),
        "Fiber Distance": float(fiber_distance),
        "No of Site": float(site_count),
    }

with right:
    st.subheader("Variable Overrides (optional)")
    # After selecting product, show cost items with variables as dropdowns
    cost_table = data[f"CostTable_{currency}"]
    rel = data["ProductCostRelation"]
    cost_items = data["CostItems"]
    related = rel[rel["ProductID"] == product_label]
    manual_variable_choices: Dict[str, str] = {}

    # In the UI section where variables are selected:
    for _, relrow in related.iterrows():
        cid = relrow["CostItemID"]
        rows = cost_table[cost_table["CostItemID"] == cid]
        
        # Get ALL variables including NA and empty strings
        raw_vars = rows["Variable"].fillna("NA").unique().tolist()
        
        # Process variables - keep NA and non-empty, convert others to string
        processed_vars = []
        for v in raw_vars:
            if pd.isna(v) or str(v).strip().upper() == "NA":
                processed_vars.append("NA")
            elif str(v).strip():
                processed_vars.append(str(v).strip())
        
        # Remove duplicates while preserving order
        seen = set()
        distinct_vars = [x for x in processed_vars if not (x in seen or seen.add(x))]
        
        # Always show selector for I06-I08 if they have any variables
        if len(distinct_vars) > 0 and (cid in ["I06","I07","I08"] or len(distinct_vars) > 1):
            item_name = cost_items.loc[cost_items["CostItemID"] == cid, "ItemName"].iloc[0]
            
            # Put NA first if exists
            if "NA" in distinct_vars:
                distinct_vars.remove("NA")
                distinct_vars.insert(0, "NA")
                
            # Default to NA if available
            default = "NA" if "NA" in distinct_vars else distinct_vars[0]
            
            choice = st.selectbox(
                f"{item_name} ({cid})",
                options=distinct_vars,
                index=distinct_vars.index(default),
                key=f"var_{cid}"
            )
            manual_variable_choices[cid] = choice

    calc_btn = st.button("🔢 Calculate", type="primary")

# ------------------------------
# Compute
# ------------------------------
if calc_btn:
    # Revenue
    TR = mrc * months + otc

    # Costs from selected product
    cost_table = data[f"CostTable_{currency}"]  # Get the correct currency table
    opex_monthly, capex_total, breakdown = compute_costs_for_product(
        product_id=product_label,
        currency_table=cost_table,
        ProductCostRelation=data["ProductCostRelation"],
        CostItems=data["CostItems"],
        user_inputs=user_inputs,
        manual_variable_choices=manual_variable_choices
    )
    regulation_fee = TR * 0.04
    

    

    # ----- DEBITDA, NPV, Discounted Payback (monthly) -----
    # Build monthly TR and OPEX series
    base_opex_monthly = opex_monthly

    # Calculate regulation fee (4% of monthly TR)
    regulation_fee_monthly = mrc * 0.04  # 4% of monthly recurring charge

    # Total monthly OPEX including regulation fee
    opex_monthly_with_fee = base_opex_monthly + regulation_fee_monthly

    otc_amount = 0.0
    regular_opex_monthly = opex_monthly

    # Check if I02 exists in the breakdown
    for item in breakdown:
        if item["CostItemID"] == "I02":
            otc_amount = item["Amount"]
            regular_opex_monthly -= otc_amount  # Remove from regular OPEX
            break
    # Monthly series generation
    TR_monthly = [mrc] * months
    OPEX_monthly_series = [regular_opex_monthly] * months

    # Apply OTC only to first month
    if months >= 1:
        TR_monthly[0] += otc  # Add OTC to first month's revenue
        OPEX_monthly_series[0] += otc_amount  # Add OTC cost to first month

    # Regulation fee (4% of each month's TR)
    regulation_fee_monthly_series = [tr * 0.04 for tr in TR_monthly]

    # Complete monthly OPEX (base + regulation fee)
    OPEX_monthly_series = [opex + reg_fee 
                        for opex, reg_fee in zip(OPEX_monthly_series, regulation_fee_monthly_series)]

    # EBITDA calculation now automatically includes regulation fee since it's in OPEX
    EBITDA_monthly = [tr - oc for tr, oc in zip(TR_monthly, OPEX_monthly_series)]


    DEBITDA_monthly = discounted_series(EBITDA_monthly, annual_discount_rate=discount_rate)
    NPV = float(np.sum(DEBITDA_monthly)) - capex_total

    cum_DEBITDA = np.cumsum(DEBITDA_monthly)

    # EBITDA uses OPEX only (CAPEX excluded)
    EBITDA_total = TR - np.sum(OPEX_monthly_series)
    EBITDA_pct = (EBITDA_total / TR * 100.0) if TR > 0 else 0.0

    # Net Cash Flow = EBITDA - CAPEX
    NetCashFlow_total = EBITDA_total - capex_total

    payback_idx = np.argmax(cum_DEBITDA - capex_total > 0) if np.any(cum_DEBITDA - capex_total > 0) else None
    discounted_payback = int(payback_idx + 1) if payback_idx is not None else None

    # ----- IRR -----
    # Build annual DEBITDA buckets from discounted monthly series
    years = int(np.ceil(months / 12))
    DEBITDA_yearly = []
    for y in range(years):
        start = y * 12
        end = min((y + 1) * 12, months)
        DEBITDA_yearly.append(float(np.sum(DEBITDA_monthly[start:end])))
    # Cashflow: t0 = -CAPEX, then yearly discounted EBITDA
    cf = [-capex_total] + DEBITDA_yearly
    irr_annual = npf.irr(cf)
    irr_annual_pct = irr_annual * 100 if irr_annual is not None else None

    # ------------------------------
    # Output
    # ------------------------------
    st.subheader("Results")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Revenue (TR)", f"{TR:,.2f} {currency}")
    colB.metric("Total Cost (OPEX × months)", f"{np.sum(OPEX_monthly_series):,.2f} {currency}")
    colC.metric("CAPEX (one-time)", f"{capex_total:,.2f} {currency}")
    colD.metric("EBITDA", f"{EBITDA_total:,.2f} {currency}")

    colA2, colB2, colC2, colD2 = st.columns(4)
    colA2.metric("Net Cash Flow (EBITDA - CAPEX)", f"{NetCashFlow_total:,.2f} {currency}")
    colB2.metric("EBITDA %", f"{EBITDA_pct:,.2f}%")
    colC2.metric("NPV (∑ DEBITDA)", f"{NPV:,.2f} {currency}")
    if irr_annual_pct is None or np.isnan(irr_annual_pct):
        colD2.metric("IRR (annual)", "N/A")
    else:
        colD2.metric("IRR (annual)", f"{irr_annual_pct:,.2f}%")

    if discounted_payback is None:
        st.info("🕒 Discounted Payback Months: Not reached within contract term.")
    else:
        st.success(f"🕒 Discounted Payback Months: {discounted_payback}")

    st.markdown("---")
    st.subheader("Breakdown")
    if breakdown:
        df_break = pd.DataFrame(breakdown)
        st.dataframe(df_break, use_container_width=True)
    else:
        st.write("No cost items found for the selected product.")

    st.markdown("---")
    st.subheader("Diagnostic (Monthly Series)")
    diag = pd.DataFrame({
        "Month": np.arange(1, months + 1),
        "TR_monthly": TR_monthly,
        "OPEX_monthly": OPEX_monthly_series,
        "EBITDA_monthly": EBITDA_monthly,
        "DEBITDA_monthly": DEBITDA_monthly,
        "Cum_DEBITDA": cum_DEBITDA,
        "Cum_DEBITDA_minus_CAPEX": cum_DEBITDA - capex_total,
    })
    st.dataframe(diag, use_container_width=True)

else:
    st.info("Select inputs and press **Calculate** to see results.")
    st.caption("Hint: Use the right column to pick Variables for cost items with multiple tiers.")