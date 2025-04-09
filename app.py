import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("AI đầu tư Bitcoin")

# Chọn mã BTC
ticker = "BTC-USD"

# Thời gian lấy dữ liệu
start = st.date_input("Từ ngày", datetime.date(2023, 1, 1))
end = st.date_input("Đến ngày", datetime.date.today())

# Tải dữ liệu từ yfinance
data = yf.download(ticker, start=start, end=end)

st.subheader("Biểu đồ giá BTC")
st.line_chart(data["Close"])

st.subheader("Dữ liệu chi tiết")
st.dataframe(data.tail())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.subheader("AI dự đoán xu hướng BTC")

# Tạo label: nếu giá ngày mai cao hơn hôm nay => 1 (tăng), ngược lại 0
data['Tomorrow Close'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow Close'] > data['Close']).astype(int)

# Tạo feature
data = data.dropna()
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
labels = data['Target']

# Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Dự đoán và đánh giá
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
st.write(f"Độ chính xác mô hình: {acc*100:.2f}%")

# Dự đoán hôm nay
latest_data = features.iloc[-1:]
prediction = model.predict(latest_data)[0]
if prediction == 1:
    st.success("AI dự đoán: Ngày mai **GIÁ TĂNG**")
else:
    st.error("AI dự đoán: Ngày mai **GIÁ GIẢM**")
