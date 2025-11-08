from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import matplotlib.pyplot as plt
import base64

app = FastAPI(title="Energy Forecast API")

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def plot_to_base64(fig):
    """Convert Matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.post("/train_arima/")
async def train_arima(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    p: int = Form(...),
    d: int = Form(...),
    q: int = Form(...),
    train_ratio: float = Form(...),
    future_steps: int = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8-sig")))

        # Detect date column
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if not date_cols:
            raise HTTPException(status_code=400, detail="No date column found")
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).set_index(date_col)

        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

        data = df[target_col].resample('H').mean().interpolate()

        split_point = int(len(data) * train_ratio)
        train, test = data[:split_point], data[split_point:]

        # Fit ARIMA
        model = ARIMA(train, order=(p, d, q))
        fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.forecast(steps=len(test))
        forecast.index = test.index

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))

        future_forecast = fitted_model.forecast(steps=future_steps)

        # Plot 1: Train vs Test vs Forecast
        fig1, ax1 = plt.subplots(figsize=(12,5))
        ax1.plot(train[-500:], label="Train", color='gray')
        ax1.plot(test, label="Actual", color='blue')
        ax1.plot(forecast, label="Forecast", color='red')
        ax1.set_title("ARIMA Forecast vs Actual")
        ax1.legend()
        plot1 = plot_to_base64(fig1)
        plt.close(fig1)

        # Plot 2: Future Forecast
        fig2, ax2 = plt.subplots(figsize=(12,4))
        ax2.plot(future_forecast, color='purple')
        ax2.set_title(f"Future Forecast ({future_steps} steps)")
        plot2 = plot_to_base64(fig2)
        plt.close(fig2)

        return {
            "mae": mae,
            "rmse": rmse,
            "forecast_plot": plot1,
            "future_forecast_plot": plot2,
            "forecast_values": forecast.tolist(),
            "future_forecast_values": future_forecast.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
