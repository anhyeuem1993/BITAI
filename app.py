import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, timedelta

st.set_page_config(page_title="AI dự đoán xu hướng BTC", layout="centered")
st.title("AI dự đoán xu hướng BTC")

# Tải và xử lý dữ liệu
@st.cache_data
def load_data():
    end = datetime.today()
    start = end - timedelta(days=365)
    data = yf.download("BTC-USD", start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Tomorrow'] = data['Close'].shift(-1)
    data
