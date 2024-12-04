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
    'AAPL': '/home//projects/AAPL.h5',
    'AMZN': '/home//projects/AMZN.h5',
    'GOOGL': '/home//projects/GOOGL.h5',
    'META': '/home//projects/META.h5',
    'MSFT': '/home//projects/MSFT.h5'
}

FEATURES = ['Close', 'Volume', 'Open', 'High', 'Low']


class StockRequest(BaseModel):
    stock_name: str


@app.post("/LSTM_Prediction")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name

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
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")

        data = yf.download(stock_name, start=start_date, end=end_date)[FEATURES]

        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data[['Close', 'Open', 'High', 'Low']] = data[['Close', 'Open', 'High', 'Low']].round(3)
        data = data.sort_values(by='Date')

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(col).strip() for col in data.columns.values]

        data.columns = data.columns.str.replace(f' {stock_name}', '', regex=False)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while loading data: {e}"
        )

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[FEATURES])

        X = []
        y = []
        for i in range(scaled_data.shape[0] - 30, scaled_data.shape[0]):
            X.append(scaled_data[i - 30:i, :])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)


        predictions = model.predict(X)

        predicted_data = np.zeros((predictions.shape[0], scaled_data.shape[1]))

        predicted_data[:, 0] = predictions.flatten()

        predicted_data = scaler.inverse_transform(predicted_data)

        predicted_prices = predicted_data[:, 0].tolist()

        return {'prediction': predicted_prices}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

