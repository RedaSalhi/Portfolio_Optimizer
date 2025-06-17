from dotenv import load_dotenv
import os
from fredapi import Fred

# Load environment variables from .env file
load_dotenv()

# Get the API key
fred_api_key = os.getenv("6c9061531b2c22864f1516a52090aabd")

# Initialize the FRED client
fred = Fred(api_key=fred_api_key)

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