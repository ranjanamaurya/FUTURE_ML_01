
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📈 AI-Powered Sales Forecasting Dashboard")

# Load data
data = pd.read_csv("sales.csv")
st.write("### Raw Sales Data", data.tail())

# Forecasting
model = Prophet()
model.fit(data.rename(columns={"ds": "ds", "y": "y"}))
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
fig1 = model.plot(forecast)
st.write("### Sales Forecast")
st.pyplot(fig1)

fig2 = model.plot_components(forecast)
st.write("### Trend & Seasonality")
st.pyplot(fig2)
