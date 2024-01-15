import tkinter as tk
from tkinter import messagebox
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ALPHA_VANTAGE_API_KEY = '8IGT66BL9IN3EUQV'

def fetch_opening_prices(stock_symbol):
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_DAILY'
    api_key = ALPHA_VANTAGE_API_KEY

    params = {
        'function': function,
        'symbol': stock_symbol,
        'apikey': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if 'Time Series (Daily)' in data:
            time_series_data = data['Time Series (Daily)']
            df = pd.DataFrame(time_series_data).T.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            return df[['Date', 'Open']]
        else:
            messagebox.showerror("Error", "Unable to fetch historical data. Please check the stock symbol.")
            return None
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        return None

def save_to_csv(data, stock_symbol):
    if data is not None:
        filename = f"{stock_symbol}_opening_prices.csv"
        data.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Opening prices saved to {filename}")

def plot_results(actual_prices, predicted_prices, forecast_steps):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, label='Actual Opening Prices', color='blue')
    ax.plot(range(len(actual_prices), len(actual_prices) + forecast_steps), predicted_prices,
            label='Predicted Opening Prices', color='red', linestyle='dashed')
    ax.set_title('Actual vs Predicted Opening Prices')
    ax.set_xlabel('Days')
    ax.set_ylabel('Opening Prices')
    ax.legend()

    return fig

def submit_action():
    stock_symbol = entry_stock.get()

    # Fetch opening prices
    opening_prices = fetch_opening_prices(stock_symbol)

    # Save data to CSV
    save_to_csv(opening_prices, stock_symbol)

    # Load data from CSV
    df = pd.read_csv(f"{stock_symbol}_opening_prices.csv")

    # Use ARIMA model
    actual_prices = df['Open'].astype(float)

    # Train ARIMA model
    order = (5, 1, 0)
    arima_model = ARIMA(actual_prices, order=order)
    arima_fit = arima_model.fit()

    # Fit GARCH model
    garch_model = arch_model(arima_fit.resid, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit()

    # Set forecast_steps after defining garch_model
    forecast_steps = 10
    
    # Forecast volatility using GARCH model
    forecast_volatility = garch_fit.conditional_volatility[-forecast_steps:]

    # Make out-of-sample predictions
    forecast_result = arima_fit.forecast(steps=forecast_steps)
    predicted_prices = forecast_result

    # Extend actual prices to include forecast period
    extended_actual_prices = np.concatenate([actual_prices, [np.nan] * forecast_steps])

    # Plot the results using Tkinter Canvas
    fig = plot_results(extended_actual_prices, predicted_prices, forecast_steps)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create the main window
window = tk.Tk()
window.title("Stock Prediction App")

# Create and place the input field
label_stock = tk.Label(window, text="Enter Stock Symbol:")
label_stock.pack(pady=10)
entry_stock = tk.Entry(window)
entry_stock.pack(pady=10)

# Create and place the submit button
submit_button = tk.Button(window, text="Enter Stock", command=submit_action)
submit_button.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
