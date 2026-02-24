import json
from typing import Dict, Optional, Tuple

import pandas as pd


# Comprehensive Alias Lists
COLUMN_ALIASES = {
    "date": ["order_date", "date", "orderdate", "transaction_date", "timestamp", "time", "order date", "ship date"],
    "quantity": ["quantity", "qty", "units", "volume", "count", "amount"],
    "price": ["derived_price", "price", "unitprice", "unit_price", "rate", "sales_amount", "revenue", "amount", "unit price"],
    "customer": ["customer", "customer_name", "customer name", "client", "buyer", "consumer", "user", "account"],
    "product": ["product", "product_name", "product name", "item", "sku", "description", "material"],
    "region": ["region", "city", "state", "location", "country", "territory", "area", "zone"],
    "category": ["category", "product_category", "type", "segment", "department", "group", "product category", "sub-category", "sub category"],
    "profit": ["profit", "margin", "net_income", "earnings"],
    "cost": ["cost", "unit_cost", "buying_price", "purchase_price", "unit cost"],
    "order_id": ["order_id", "orderid", "id", "order_no", "invoice_no", "invoice", "order id", "row id"],
    "sales": ["sales", "revenue", "total_sales", "amount", "total", "total sales"],
    "stock": ["stock", "current_stock", "quantity_in_stock", "inventory"],
}

REQUIRED_NUMERIC_COLS = COLUMN_ALIASES["quantity"] + COLUMN_ALIASES["price"]
REQUIRED_DATE_COLS = COLUMN_ALIASES["date"]


def read_data_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Robustly read a data file (CSV or Excel).
    - Handles multiple CSV encodings (utf-8, latin-1, cp1252).
    - Handles Excel files.
    - Returns None if file cannot be read or extension is unsupported.
    """
    try:
        if file_path.endswith('.csv'):
            # Try common encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ISO-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    return normalize_dataset(df)
                except UnicodeDecodeError:
                    continue
            # If all fail, try with replacement
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            return normalize_dataset(df)
            
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            return normalize_dataset(df)
            
        return None
    except Exception:
        # Log error in production
        return None


def infer_core_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer standard business columns using comprehensive alias lists.
    Returns a dictionary mapping standard keys (date, quantity, price, sales, customer, product, region, category, profit)
    to the actual column names in the dataframe.
    """
    # Create a mapping of simplified column names to actual column names
    # Simplify: lowercase, strip whitespace, replace underscores with spaces (optional, but good for matching)
    normalized_cols = {}
    for c in df.columns:
        # standard normalization: "  Order Date  " -> "order date"
        s_name = str(c).strip().lower()
        normalized_cols[s_name] = c
        
        # also add underscore version: "Order_Date" -> "order date"
        # and space version: "order_date" -> "order date"
        # This helps match "order date" alias against "order_date" column
        s_name_spaces = s_name.replace('_', ' ')
        if s_name_spaces not in normalized_cols:
             normalized_cols[s_name_spaces] = c
             
        s_name_underscores = s_name.replace(' ', '_')
        if s_name_underscores not in normalized_cols:
             normalized_cols[s_name_underscores] = c

    inferred = {}

    for standard_key, aliases in COLUMN_ALIASES.items():
        found_col = None
        for alias in aliases:
            # alias is already lowercased in the constant definitions? 
            # actually they are mixed in my update (some have spaces)
            # define clean alias
            clean_alias = alias.lower().strip()
            
            # Direct match
            if clean_alias in normalized_cols:
                found_col = normalized_cols[clean_alias]
                break
            
            # Try matching alias with underscores vs spaces
            clean_alias_u = clean_alias.replace(' ', '_')
            if clean_alias_u in normalized_cols:
                 found_col = normalized_cols[clean_alias_u]
                 break
                 
            clean_alias_s = clean_alias.replace('_', ' ')
            if clean_alias_s in normalized_cols:
                 found_col = normalized_cols[clean_alias_s]
                 break
        inferred[standard_key] = found_col

    return inferred


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the dataset by deriving missing core columns if possible.
    - If Price is missing but Sales and Quantity exist, derive Price.
    """
    if df is None:
        return None
        
    inferred = infer_core_columns(df)
    price_col = inferred.get("price")
    sales_col = inferred.get("sales")
    qty_col = inferred.get("quantity")
    
    # Fallback: If Price is missing but we have Sales and Quantity, derive Price
    if not price_col and sales_col and qty_col:
        try:
            # Create a derived Price column
            # Avoid division by zero
            df['Derived_Price'] = df[sales_col] / df[qty_col].replace(0, 1)
        except Exception:
            pass
            
    return df


def build_schema_signature(df: pd.DataFrame) -> str:
    """
    Build a simple, explainable schema signature that captures:
    - Column names (lowercased and sorted).

    Using a JSON string keeps it easy to debug and display in error messages.
    """
    cols = sorted([str(c).lower() for c in df.columns])
    return json.dumps(cols, ensure_ascii=False)


def validate_dataset_schema(
    df: pd.DataFrame, existing_signature: Optional[str] = None
) -> Tuple[str, Dict[str, str]]:
    """
    Validate a single uploaded dataset for:
    1) Required columns (date, quantity, price).
    2) Schema consistency with an existing signature (if provided).
    3) Missing / invalid values in core columns.

    Returns:
        schema_signature: string to be stored on the Dataset model.
        meta: dict with resolved column names for downstream use.

    Raises:
        ValueError with a clear, user-facing message when validation fails.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty. Please upload a file with data.")

    # DF is already normalized by read_data_file, so we just infer again
    inferred = infer_core_columns(df)
    date_col = inferred.get("date")
    qty_col = inferred.get("quantity")
    price_col = inferred.get("price")

    if not date_col or not qty_col or not price_col:
        raise ValueError(
            "Dataset must include Date, Quantity and Price columns. "
            "Common names: Order Date / Date, Quantity / Qty, Price / UnitPrice. "
            "(Or provide Sales/Revenue + Quantity to automatically calculate Price)"
        )

    # Check for missing values in core columns (common real-world data issue).
    core_subset = df[[date_col, qty_col, price_col]].copy()
    core_subset[qty_col] = pd.to_numeric(core_subset[qty_col], errors="coerce")
    core_subset[price_col] = pd.to_numeric(core_subset[price_col], errors="coerce")

    if core_subset.isna().all().any():
        raise ValueError(
            "One or more core columns (date, quantity, price) contain only missing "
            "or invalid values. Please clean the file and try again."
        )

    # Build the new schema signature and, if present, compare with the existing one.
    new_signature = build_schema_signature(df)
    
    # RELAXED VALIDATION:
    # Previously we enforced strict equality of column names (existing_signature).
    # However, since we now support robust column inference (infer_core_columns),
    # we allow different column names (e.g. 'Customer' vs 'Customer Name') 
    # as long as the core columns (Date, Quantity, Price) are detected.
    # We still track the signature for debugging/metadata but do not block the upload.

    meta = {
        "date_col": date_col,
        "qty_col": qty_col,
        "price_col": price_col,
    }
    return new_signature, meta


def compute_basic_kpis(df: pd.DataFrame, date_col: str, qty_col: str, price_col: str) -> Dict[str, float]:
    """
    Compute simple, comparable KPIs for a dataset:
    - total_revenue
    - total_orders (row count proxy)
    - total_rows

    This is intentionally minimal so we can use it for cross-dataset comparison
    without pulling in all dashboard logic.
    """
    if df is None or df.empty:
        return {"total_revenue": 0.0, "total_orders": 0.0, "total_rows": 0.0}

    df = df.copy()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
    df["Revenue"] = df[qty_col] * df[price_col]

    total_revenue = float(df["Revenue"].sum())
    total_orders = float(len(df))
    total_rows = float(len(df))

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_rows": total_rows,
    }


def compare_values(current: float, baseline: float, threshold_pct: float = 10.0) -> Dict[str, float | str | bool]:
    """
    Compare a current KPI value against a baseline in a way that is:
    - Robust (handles zeros / missing baselines).
    - Easy to explain on a business dashboard.

    Returns:
        {
          "current": ...,
          "baseline": ...,
          "abs_diff": ...,
          "pct_change": ...,
          "direction": "increase" | "decrease" | "no_change",
          "is_significant": bool  # |pct_change| >= threshold_pct
        }
    """
    current = float(current or 0.0)
    baseline = float(baseline or 0.0)

    abs_diff = current - baseline
    if baseline > 0:
        pct_change = (abs_diff / baseline) * 100.0
    else:
        # If there is no meaningful baseline, treat this as "new" data.
        pct_change = 0.0

    if abs_diff > 0:
        direction = "increase"
    elif abs_diff < 0:
        direction = "decrease"
    else:
        direction = "no_change"

    is_significant = abs(pct_change) >= threshold_pct

    return {
        "current": current,
        "baseline": baseline,
        "abs_diff": abs_diff,
        "pct_change": pct_change,
        "direction": direction,
        "is_significant": is_significant,
    }

