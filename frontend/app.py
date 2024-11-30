import streamlit as st
import requests
import plotly.graph_objs as go
import pandas as pd
from datetime import date

API_URL = "http://127.0.0.1:8000/LSTM_Prediction"

st.title('Stock Price Prediction App')

stocks = ['', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT']
stock_name = st.selectbox('Choose a stock name', stocks)

if st.button('Predict'):
    if not stock_name:
        st.error('Please select a stock name.')
    else:
        payload = {
            "stock_name": stock_name
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            predictions = response.json()
            
            if "prediction" in predictions:
                predicted_prices = predictions["prediction"]
                
                predicted_dates = pd.date_range(
                    start=pd.Timestamp.now() + pd.Timedelta(days=1), 
                    periods=len(predicted_prices), 
                    freq='B'
                ).strftime("%Y-%m-%d").tolist()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=predicted_dates, 
                    y=predicted_prices, 
                    mode='lines+markers', 
                    name='Predicted Price',
                    line=dict(color='white', width=2)
                ))
                
                fig.update_layout(
                    title=f"{stock_name} Predicted Prices",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    height=500,
                    width=800,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
        except ValueError as e:
            st.error(f"Error parsing API response: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
