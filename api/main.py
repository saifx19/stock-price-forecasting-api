from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = FastAPI()

MODEL_PATHS = {
    'AAPL': '../models/AAPL.h5',
    'AMZN': '../models/AMZN.h5',
    'GOOGL': '../models/GOOGL.h5',
    'META': '../models/META.h5',
    'MSFT': '../models/MSFT.h5'
}

# Define the features to be used for processing
FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low']


class StockRequest(BaseModel):
    stock_name: str


@app.post("/LSTM_Prediction")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name

    # Load the saved model based on stock name
    try:
        model = load_model(MODEL_PATHS[stock_name])
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Model for stock '{stock_name}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {e}"
        )

    try:
        # Calculate the date range
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Download the data from Yahoo Finance
        data = yf.download(stock_name, start=start_date, end=end_date)[FEATURES]

        # Reset the index and clean up the data
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data[['Close', 'Open', 'High', 'Low']] = data[['Close', 'Open', 'High', 'Low']].round(3)
        data = data.sort_values(by='Date')

        # Flatten columns if MultiIndex exists
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(col).strip() for col in data.columns.values]

        # Clean column names
        data.columns = data.columns.str.replace(f' {stock_name}', '', regex=False)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading data: {e}"
        )

    try:
        # Apply scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[FEATURES])

        # Prepare the input data for prediction
        X = []
        y = []
        for i in range(scaled_data.shape[0] - 30, scaled_data.shape[0]):
            X.append(scaled_data[i - 30:i, :])  # Use all features as input
            y.append(scaled_data[i, 0])  # Use the Close column as target

        X = np.array(X)
        y = np.array(y)


        # Make predictions
        predictions = model.predict(X)

        # Prepare a placeholder array for predictions
        # The first dimension matches predictions; the second dimension matches scaled_data columns
        predicted_data = np.zeros((predictions.shape[0], scaled_data.shape[1]))

        # Fill predictions in the first column
        predicted_data[:, 0] = predictions.flatten()

        # Inverse transform the predictions
        predicted_data = scaler.inverse_transform(predicted_data)

        # Convert predictions to a list of prices
        predicted_prices = predicted_data[:, 0].tolist()

        return {'prediction': predicted_prices}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )
