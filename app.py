import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(page_title="AI dự đoán xu hướng BTC", layout="wide")
st.title("AI dự đoán xu hướng BTC")

# Tải dữ liệu BTC từ Yahoo Finance
@st.cache_data
def load_data():
    end = datetime.today()
    start = end - timedelta(days=365)
    data = yf.download("BTC-USD", start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Tomorrow'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data

data = load_data()

# Hiển thị dữ liệu
st.subheader("Dữ liệu giá BTC")
st.dataframe(data.tail())

# Tạo biến mục tiêu và đặc trưng
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Dự đoán xu hướng cho ngày hiện tại
latest = data.iloc[-1][features].values.reshape(1, -1)
pred = model.predict(latest)[0]
prob = model.predict_proba(latest)[0][1]

st.subheader("Dự đoán xu hướng tiếp theo")
if pred == 1:
    st.success(f"Dự đoán: Giá BTC sẽ **TĂNG** với xác suất {prob:.2%}")
else:
    st.error(f"Dự đoán: Giá BTC sẽ **GIẢM** với xác suất {1 - prob:.2%}")
