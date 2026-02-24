
import pandas as pd
from datetime import datetime, timedelta

def test_filtering_logic():
    print("creating test data...")
    # Create dataset with dates from 1 year ago
    today = datetime.now()
    dates = [today - timedelta(days=365 + i) for i in range(10)]
    # Add one recent date (today) to verify mixed data behavior if needed, 
    # but for this test we simulate a fully historical dataset.
    
    # Pure historical data (e.g. from 2025 if now is 2026, or 2023 if now is 2024)
    # Let's say user uploaded a file from Jan 2024.
    last_year = today - timedelta(days=365)
    dates_historical = [last_year - timedelta(days=i) for i in range(10)]
    
    df = pd.DataFrame({
        'date': dates_historical,
        'val': range(10)
    })
    
    print(f"Dataset Max Date: {df['date'].max()}")
    print(f"Current System Time: {pd.Timestamp.now()}")
    
    # -------------------------------------------------------------
    # Logic from views.py
    # -------------------------------------------------------------
    date_range = '30' # Last 30 days
    
    # Current Implementation: Relative to NOW
    cutoff_date_now = pd.Timestamp.now() - pd.Timedelta(days=30)
    df_filtered_now = df[df['date'] >= cutoff_date_now]
    
    print(f"\nFilter 'Last 30 Days' (Relative to NOW):")
    print(f"Cutoff: {cutoff_date_now}")
    print(f"Rows found: {len(df_filtered_now)}")
    
    # Proposed Implementation: Relative to MAX DATE in dataset
    max_date = df['date'].max()
    cutoff_date_relative = max_date - pd.Timedelta(days=30)
    df_filtered_relative = df[df['date'] >= cutoff_date_relative]
    
    print(f"\nFilter 'Last 30 Days' (Relative to Max Date):")
    print(f"Cutoff: {cutoff_date_relative}")
    print(f"Rows found: {len(df_filtered_relative)}")
    
    if len(df_filtered_now) == 0 and len(df_filtered_relative) > 0:
        print("\n[CONFIRMED] Current logic returns 0 rows for historical data. Proposed logic works.")
    else:
        print("\n[INCONCLUSIVE] Could not reproduce strict zero-row issue with this setup.")

if __name__ == "__main__":
    test_filtering_logic()
