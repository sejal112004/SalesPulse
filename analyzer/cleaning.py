import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Applies the cleaning process and returns (df_clean, report_log).
    report_log is a list of strings describing changes.
    """
    if df is None or df.empty:
        return df, []

    df = df.copy()
    report_log = []
    original_rows = len(df)

    # 1. Inspect & Standardize Headers
    df.columns = df.columns.str.strip()
    
    # Identify key columns
    lower_cols = {c.lower(): c for c in df.columns}
    date_col = lower_cols.get("order_date") or lower_cols.get("date") or lower_cols.get("orderdate") 
    qty_col = lower_cols.get("quantity") or lower_cols.get("qty")
    price_col = lower_cols.get("price") or lower_cols.get("unitprice") or lower_cols.get("unit_price")
    
    # 3. Remove duplicates
    df = df.drop_duplicates()
    dupes_removed = original_rows - len(df)
    if dupes_removed > 0:
        report_log.append(f"Removed {dupes_removed} duplicate rows.")

    # 4. Convert Data Types
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        prev_len = len(df)
        df = df.dropna(subset=[date_col])
        invalid_dates = prev_len - len(df)
        if invalid_dates > 0:
            report_log.append(f"Removed {invalid_dates} rows with invalid dates in column '{date_col}'.")

    # Convert numeric columns
    numeric_candidates = [qty_col, price_col, 'sales', 'revenue', 'cost', 'profit']
    for col in numeric_candidates:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 5. Handle Outliers (IQR)
    for col in numeric_cols:
        if col == date_col: continue
        
        # Clip negative
        if col in [qty_col, price_col]:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df[col] = df[col].clip(lower=0)
                report_log.append(f"Corrected {neg_count} negative values in '{col}'.")

        # IQR Capping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] > upper_bound)).sum()
        if outliers > 0:
            df[col] = df[col].clip(upper=upper_bound)
            report_log.append(f"Capped {outliers} outliers in '{col}' to upper limit {round(upper_bound, 2)}.")

    # 2. Handle Missing Values
    # Numeric
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            df[col] = df[col].fillna(fill_val)
            report_log.append(f"Filled {missing} missing values in '{col}' with median ({round(fill_val, 2)}).")

    # Categorical
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_series = df[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report_log.append(f"Filled {missing} missing values in '{col}' with '{fill_val}'.")

    # 6. Standardize Text
    for col in categorical_cols:
        if col != date_col:
            # Check if any changes happen (simplified check)
            # Just do it
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({'Nan': 'Unknown', 'None': 'Unknown', 'Null': 'Unknown'})

    # 7. Derived Columns
    if qty_col and price_col:
        # Check if Revenue exists and needs filling?
        # Just overwrite or fillna?
        # Let's say we calculated it.
        revenue_values = df[qty_col] * df[price_col]
        if 'Revenue' not in df.columns:
            df['Revenue'] = revenue_values
            report_log.append("Created derived column 'Revenue' (Quantity * Price).")
        else:
            # If it had missing, we probably filled it above. 
            pass

    if date_col:
        if 'Month' not in df.columns:
            df['Month'] = df[date_col].dt.month_name()
            report_log.append("Created derived column 'Month'.")
        if 'Year' not in df.columns:
            df['Year'] = df[date_col].dt.year
            report_log.append("Created derived column 'Year'.")

    # Final Catch-All
    remaining_na = df.isna().sum().sum()
    if remaining_na > 0:
        df = df.fillna("dd")
        report_log.append(f"Filled {remaining_na} remaining empty cells with 'dd'.")
    
    # Empty strings
    # difficult to count efficiently without slow apply, just do it.
    df = df.replace(r'^\s*$', 'dd', regex=True)

    if not report_log:
        report_log.append("No issues found. Data is clean.")

    return df, report_log
