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
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)  # Tránh lỗi ValueError khi so sánh
    return data

data = load_data()

# Thêm cột mục tiêu
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

# Tính toán thêm các đặc trưng
data['Change'] = data['Close'] - data['Open']
data['High_Low'] = data['High'] - data['Low']
data['Volatility'] = (data['High'] - data['Low']) / data['Open']

# Các đặc trưng đầu vào cho mô hình
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'High_Low', 'Volatility']
X = data[features]
y = data['Target']

# Tách dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên dữ liệu mới nhất
latest_data = X.iloc[[-1]]
prediction = model.predict(latest_data)[0]
prob = model.predict_proba(latest_data)[0]

# Hiển thị kết quả dự đoán
st.subheader("Kết quả dự đoán hôm nay:")
if prediction == 1:
    st.success(f"AI dự đoán: **Giá BTC sẽ TĂNG** với xác suất {prob[1]*100:.2f}%")
else:
    st.error(f"AI dự đoán: **Giá BTC sẽ GIẢM** với xác suất {prob[0]*100:.2f}%")

# Thống kê mô hình
st.subheader("Đánh giá mô hình trên tập kiểm tra:")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
