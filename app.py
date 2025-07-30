
import requests
import json
import time
import pandas as pd
from IPython.display import display, HTML

# --- Configuration ---
# IMPORTANT: Leave apiKey as an empty string. Jupyter environment will provide it at runtime.
API_KEY = ""
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Helper Functions ---

def call_gemini_api(prompt, retries=3, delay=1.0):
    """
    Calls the Gemini API with exponential backoff.

    Args:
        prompt (str): The text prompt for the LLM.
        retries (int): Number of retries remaining.
        delay (float): Current delay in seconds for exponential backoff.

    Returns:
        dict or None: Parsed JSON response from the API, or None on failure.
    """
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "month": {"type": "STRING"},
                        "predictedSales": {"type": "NUMBER"}
                    },
                    "propertyOrdering": ["month", "predictedSales"]
                }
            }
        }
    }

    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and \
           result['candidates'][0]['content']['parts'][0].get('text'):
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_string)
        else:
            print("Error: Invalid API response structure.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        if retries > 0:
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            return call_gemini_api(prompt, retries - 1, delay * 2) # Exponential backoff
        else:
            print("Max retries reached. Could not get a response from the API.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response text: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def forecast_sales(historical_sales: list[float], forecast_months: int) -> pd.DataFrame:
    """
    Generates sales forecast using the Gemini API and returns it as a pandas DataFrame.

    Args:
        historical_sales (list[float]): List of historical monthly sales figures.
        forecast_months (int): Number of months to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecasted sales, or an empty DataFrame on failure.
    """
    if not historical_sales:
        print("Error: Historical sales data cannot be empty.")
        return pd.DataFrame()
    if not 1 <= forecast_months <= 12:
        print("Error: Number of forecast months must be between 1 and 12.")
        return pd.DataFrame()

    # Construct the prompt for the AI model
    prompt = f"""Given the following historical monthly sales data (in USD): {', '.join(map(str, historical_sales))}.
    Please forecast sales for the next {forecast_months} months.
    Provide the forecast as a JSON array of objects, where each object has 'month' (e.g., 'Month 1', 'Month 2') and 'predictedSales' (a number).
    Ensure 'predictedSales' is a numerical value.
    Do not include any other text in your response, only the JSON."""

    print("Generating forecast... Please wait.")
    forecast_data = call_gemini_api(prompt)

    if forecast_data:
        df = pd.DataFrame(forecast_data)
        # Format predicted sales for better readability
        df['predictedSales'] = df['predictedSales'].apply(lambda x: f"${x:,.2f}")
        print("\n--- Sales Forecast ---")
        return df
    else:
        print("Failed to get a valid forecast from the AI.")
        return pd.DataFrame()

# --- How to Use in Jupyter Notebook ---

# 1. Provide your historical sales data
# Example: Sales for 12 months
historical_data = [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1450, 1600]

# 2. Specify the number of months you want to forecast
months_to_forecast = 6

# 3. Call the forecasting function
forecast_df = forecast_sales(historical_data, months_to_forecast)

# 4. Display the results (Jupyter will automatically render DataFrames)
if not forecast_df.empty:
    display(forecast_df)

# You can also try different inputs:
# historical_data_2 = [50, 55, 60, 58, 65]
# months_to_forecast_2 = 3
# forecast_df_2 = forecast_sales(historical_data_2, months_to_forecast_2)
# if not forecast_df_2.empty:
#     display(forecast_df_2)
