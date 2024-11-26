import streamlit as st
import requests
import plotly.graph_objs as go
import pandas as pd
from datetime import date

# API URL
API_URL = "http://127.0.0.1:8000/LSTM_Prediction"

# Title of the app
st.title('Stock Price Prediction App')

# Dropdown menu for stock selection
stocks = ['', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT']
stock_name = st.selectbox('Choose a stock name', stocks)

# Predict button functionality
if st.button('Predict'):
    # Validate stock name input
    if not stock_name:
        st.error('Please select a stock name.')
    else:
        # Prepare payload for API request
        payload = {
            "stock_name": stock_name
        }
        
        try:
            # Send request to API
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Check for HTTP request errors
            
            # Parse JSON response
            predictions = response.json()
            
            # Validate response structure
            if "prediction" in predictions:
                predicted_prices = predictions["prediction"]
                
                # Generate business day dates for predictions
                predicted_dates = pd.date_range(
                    start=pd.Timestamp.now() + pd.Timedelta(days=1), 
                    periods=len(predicted_prices), 
                    freq='B'
                ).strftime("%Y-%m-%d").tolist()
                
                # Create interactive Plotly figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=predicted_dates, 
                    y=predicted_prices, 
                    mode='lines+markers', 
                    name='Predicted Price',
                    line=dict(color='white', width=2)
                ))
                
                # Customize plot layout
                fig.update_layout(
                    title=f"{stock_name} Predicted Prices",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    height=500,
                    width=800,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            
        
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
        except ValueError as e:
            st.error(f"Error parsing API response: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
                st.error("Unexpected response format.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request: {e}")
