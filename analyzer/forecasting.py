import pandas as pd
import numpy as np
from datetime import timedelta

def preprocess_for_forecasting(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Preprocesses the dataframe for forecasting:
    - Parses dates
    - Sorts by date
    - Aggregates duplicates
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Sort and aggregate duplicates (sum revenue for same day/time)
    df = df.groupby(date_col)[value_col].sum().reset_index()
    df = df.sort_values(date_col)
    
    return df

def aggregate_data(df: pd.DataFrame, date_col: str, value_col: str, period: str = 'M'):
    """
    Aggregates data by period (M=Monthly, Q=Quarterly, Y=Yearly).
    Returns DataFrame with 'period_dt' (datetime) and 'y' (value).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['period_dt', 'y', 'label'])

    df = df.copy()
    
    if period == 'M':
        # Resample to Month Start
        df = df.set_index(date_col).resample('MS')[value_col].sum().reset_index()
        df['label'] = df[date_col].dt.strftime('%b %Y') # Jan 2024
    elif period == 'Q':
        # Resample to Quarter Start
        df = df.set_index(date_col).resample('QS')[value_col].sum().reset_index()
        df['label'] = df[date_col].apply(lambda x: f"Q{(x.month-1)//3 + 1} {x.year}") # Q1 2024
    elif period == 'Y':
        # Resample to Year Start
        df = df.set_index(date_col).resample('YS')[value_col].sum().reset_index()
        df['label'] = df[date_col].dt.strftime('%Y') # 2024
    else:
        # Default to Monthly
        df = df.set_index(date_col).resample('MS')[value_col].sum().reset_index()
        df['label'] = df[date_col].dt.strftime('%b %Y')

    df.rename(columns={date_col: 'period_dt', value_col: 'y'}, inplace=True)
    # Filter out periods with 0 revenue if they are leading (keep internal zeros? maybe, but for sales often 0 means no data)
    # For linear regression, continuous time is better. We keep 0s.
    return df

def linear_regression_forecast(df: pd.DataFrame, periods_to_forecast: int = 3, freq: str = 'M'):
    """
    Performs specific time-series linear regression.
    params:
      df: DataFrame with ['period_dt', 'y']
      periods_to_forecast: number of future periods
      freq: 'M', 'Q', 'Y' (pandas offset aliases)
    
    Returns:
      history_df: ['period_dt', 'y', 'trend', 'label'] (actuals + trend line)
      forecast_df: ['period_dt', 'yhat', 'label'] (future predictions)
    """
    if df is None or len(df) < 2:
        return df, pd.DataFrame()

    # Create an integer time index (0, 1, 2, ...) for regression
    df = df.copy()
    df['t'] = range(len(df))
    
    # Linear Regression: y = mx + c
    # x = t, y = revenue
    x = df['t'].values
    y = df['y'].values
    
    # Calculate slope (m) and intercept (c)
    # N * sum(xy) - sum(x)*sum(y) / ... standard formula or numpy
    # Numpy polyfit is robust and simple (deg=1 is linear)
    m, c = np.polyfit(x, y, 1)
    
    # Calculate R-squared
    y_pred_hist = m * x + c
    ss_res = np.sum((y - y_pred_hist) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate Standard Error of the Estimate (SEE)
    n = len(df)
    if n > 2:
        see = np.sqrt(ss_res / (n - 2))
    else:
        see = 0
        
    # Calculate Prediction Interval (approximate 95% CI)
    # interval = t_crit * see * sqrt(1 + 1/n + (x_new - x_bar)^2 / sum(x - x_bar)^2)
    # Simplified for visualization: +/- 1.96 * see (assuming large enough sample, normal dist)
    # We will use t-distribution critical value for small samples ideally, but 1.96 is okay for rough viz
    # stricter: use 2 * see
    
    margin_of_error = 2 * see
    
    # Check if trend is significant (simple heuristic)
    # If r_squared is very low, the trend might be noise.
    # We return the metrics for the view to decide how to display.
    
    df['trend'] = y_pred_hist
    df['lower_ci'] = df['trend'] - margin_of_error
    df['upper_ci'] = df['trend'] + margin_of_error
    
    # Generate future periods
    last_date = df['period_dt'].max()
    future_dates = []
    
    # Determine next dates based on freq
    # mapping freq to offset
    offset_map = {
        'M': pd.DateOffset(months=1),
        'Q': pd.DateOffset(months=3),
        'Y': pd.DateOffset(years=1)
    }
    offset = offset_map.get(freq, pd.DateOffset(months=1))
    
    for i in range(1, periods_to_forecast + 1):
        future_dates.append(last_date + (offset * i))
        
    forecast_df = pd.DataFrame({'period_dt': future_dates})
    
    # Predict future values
    # Future t starts from len(df)
    future_t = np.arange(len(df), len(df) + periods_to_forecast)
    forecast_df['yhat'] = m * future_t + c
    
    # Add Confidence Intervals for future
    # Uncertainty grows with time? 
    # For simple linear regression visualizations, constant width or "bowtie" shape.
    # We'll implementation "bowtie" widening slightly for future:
    # SE_forecast = see * sqrt(1 + 1/n + (x_future - x_mean)^2 / Sxx)
    
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)
    
    def get_ci_width(t_val, see, n, mean, sxx):
        if n <= 2 or sxx == 0:
            return 0
        # t-statistic approx 2 for 95%
        return 2 * see * np.sqrt(1 + (1/n) + ((t_val - mean)**2 / sxx))

    ci_widths = [get_ci_width(t, see, n, x_mean, sxx) for t in future_t]
    
    forecast_df['lower_ci'] = forecast_df['yhat'] - ci_widths
    forecast_df['upper_ci'] = forecast_df['yhat'] + ci_widths
    
    # Avoid negative forecasts for sales
    forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
    forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(lower=0)
    # Upper CI doesn't need clipping usually
    
    # Add labels
    if freq == 'M':
        forecast_df['label'] = forecast_df['period_dt'].dt.strftime('%b %Y')
    elif freq == 'Q':
        forecast_df['label'] = forecast_df['period_dt'].apply(lambda x: f"Q{(x.month-1)//3 + 1} {x.year}")
    elif freq == 'Y':
        forecast_df['label'] = forecast_df['period_dt'].dt.strftime('%Y')
        
    metrics = {
        'r_squared': r_squared,
        'slope': m,
        'intercept': c,
        'standard_error': see
    }
        
    return df, forecast_df, metrics
