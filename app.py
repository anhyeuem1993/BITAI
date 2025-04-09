import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(page_title="AI dự đoán xu hướng BTC", layout="wide")
st.title("AI dự đoán xu hướng BTC")

@st.cache_data
def load_data():
    end = datetime.today()
    start = end - timedelta(days=365)
    data = yf.download("BTC-USD", start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # chỉ giữ lại các cột cần thiết
    data['Tomorrow'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data

data = load_data()

# Kiểm tra các cột có tồn tại không
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Tomorrow']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Các cột bị thiếu trong dữ liệu: {missing_columns}")
    st.stop()

# Hiển thị dữ liệu
st.subheader("Dữ liệu giá BTC")
st.dataframe(data.tail())

# Chuẩn bị dữ liệu học
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
features = ['Open', 'High', 'Low', 'Close', 'Volume']

try:
    X = data[features]
    y = data['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Dự đoán ngày hôm sau
    latest = data.iloc[-1][features].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][1]

    st.subheader("Dự đoán xu hướng tiếp theo")
    if pred == 1:
        st.success(f"Dự đoán: Giá BTC sẽ **TĂNG** với xác suất {prob:.2%}")
    else:
        st.error(f"Dự đoán: Giá BTC sẽ **GIẢM** với xác suất {1 - prob:.2%}")

except Exception as e:
    st.error(f"Lỗi khi xử lý dữ liệu hoặc huấn luyện mô hình: {e}")
    st.write("Các cột hiện có trong data:", data.columns.tolist())
    st.dataframe(data.tail())
