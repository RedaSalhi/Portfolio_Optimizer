from dotenv import load_dotenv
import os
from fredapi import Fred

load_dotenv()

fred_api_key = os.getenv("6c9061531b2c22864f1516a52090aabd")

fred = Fred(api_key='6c9061531b2c22864f1516a52090aabd')

def get_latest_risk_free_rate():
    """
    Fetch the latest available 3-Month Treasury Bill rate (DGS3MO) from FRED.
    Returns:
        float: The latest risk-free rate as a percentage (e.g., 5.12 for 5.12%).
    """
    series_id = 'DGS3MO'
    data = fred.get_series(series_id)
    latest_value = data.dropna().iloc[-1]  # Get the most recent non-NaN value
    return latest_value 