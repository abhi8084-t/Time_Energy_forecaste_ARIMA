# Energy Forecasting Project: ARIMA-based Energy Consumption Prediction

## 1. Project Overview  
This project implements an **Energy Consumption Forecasting System** using a combination of a backend API (built with FastAPI) and a frontend application (built with Streamlit).  
The system allows users to upload a CSV file containing timestamped energy usage and related features, select a target numeric column, choose model parameters (ARIMA: p, d, q plus train/test split and future steps) and then get:  
- Model training metrics: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)  
- A plot of actual vs forecasted values  
- A plot of future forecast values  
- Forecast data values (for further use)  

The core forecasting model uses the ARIMA (AutoRegressive Integrated Moving Average) algorithm, making this setup suitable for **time series forecasting** of energy consumption.

---

## 2. Dataset Description  
This project uses as its reference the dataset: **Appliances Energy Prediction** from the UCI Machine Learning Repository.  
- Dataset name: [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)  
- Donated on: February 14, 2017. :contentReference[oaicite:4]{index=4}  
- Instances: 19,735 (10 minute intervals for ~4.5 months) :contentReference[oaicite:5]{index=5}  
- Features: 28 variables (time‑series + environmental + random variables) :contentReference[oaicite:6]{index=6}  
- Characteristics: Multivariate, real‑valued, time series regression. :contentReference[oaicite:7]{index=7}  
- No missing values. :contentReference[oaicite:8]{index=8}  
- License: Creative Commons Attribution 4.0 International (CC BY 4.0) :contentReference[oaicite:9]{index=9}  

### Key Variables  
| Variable       | Type      | Description                                                                                                                                  |
|----------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `date`         | Time‑Stamp| Timestamp of each 10‑minute interval. :contentReference[oaicite:10]{index=10}                                                                            |
| `Appliances`   | Integer   | Energy use of appliances in Wh (Target) :contentReference[oaicite:11]{index=11}                                                                        |
| `lights`       | Integer   | Energy use of light fixtures in the house in Wh. :contentReference[oaicite:12]{index=12}                                                                |
| `T1, T2, …`    | Continuous| Temperatures in different rooms/areas in Celsius. :contentReference[oaicite:13]{index=13}                                                                |
| `RH_1, RH_2,…` | Continuous| Humidity measurements in different rooms/areas in %. :contentReference[oaicite:14]{index=14}                                                              |
| `T_out`, `RH_out`, `Press_mm_hg`, etc. | Continuous | Weather variables from a nearby weather station (outside house) merged with internal data. :contentReference[oaicite:15]{index=15} |
| `rv1`, `rv2`   | Random    | Two random variables included to test model filtering of non‑predictive features. :contentReference[oaicite:16]{index=16}                                 |

### Why this dataset?  
- Real‑world energy consumption data of a low‑energy building, combined with environmental & weather variables.  
- Good size (~19,700 rows) and high frequency (10‑minute intervals) allowing meaningful time‑series forecasting experiments.  
- Well‑documented and publicly available under a liberal license (CC BY 4.0).

---

## 3. Project Structure  

### Backend (FastAPI)
```bash
pip install -r requirements.txt
uvicorn main:app --reload

The API will run at http://127.0.0.1:8000.

Frontend (Streamlit)
streamlit run frontend.py


The Streamlit app runs at http://localhost:8501.

