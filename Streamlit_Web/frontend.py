import streamlit as st
import requests
import pandas as pd
import base64

API_URL = "http://127.0.0.1:8000/train_arima/"

st.set_page_config(page_title="Energy Forecasting (ARIMA API)", layout="wide")
st.title("⚡ Energy Consumption Forecasting (ARIMA Model via API)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Preview data
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Data Preview")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Target column
# Automatically detect numeric columns (exclude datetime)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in CSV for forecasting")
        st.stop()

    target_col = st.selectbox("Select Target Column (numeric only)", numeric_cols)

    # ARIMA parameters
    st.subheader("ARIMA Parameters")
    p = st.number_input("p (AR)", min_value=0, max_value=5, value=1)
    d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (MA)", min_value=0, max_value=5, value=1)

    train_ratio = st.slider("Train-Test Split Ratio", 0.6, 0.9, 0.8)
    future_steps = st.slider("Future Forecast Steps", 24, 168, 48)

    if st.button("Train ARIMA via API"):
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Prepare files correctly
        files = {
            "file": (uploaded_file.name, uploaded_file, "text/csv")
        }
        
        # Form data
        data = {
            "target_col": target_col,
            "p": p,
            "d": d,
            "q": q,
            "train_ratio": train_ratio,
            "future_steps": future_steps
        }
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display metrics
                st.subheader("📊 Model Metrics")
                st.write(f"**MAE:** {result['mae']:.2f}")
                st.write(f"**RMSE:** {result['rmse']:.2f}")

                # Forecast vs Actual
                st.subheader("Forecast vs Actual")
                st.image(base64.b64decode(result['forecast_plot']))

                # Future forecast
                st.subheader(f"Future Forecast ({future_steps} steps)")
                st.image(base64.b64decode(result['future_forecast_plot']))
            else:
                st.error(f"❌ API request failed! {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("👈 Upload a CSV file to start")
