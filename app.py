import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.title("AI dự đoán xu hướng BTC")

# Tải dữ liệu
@st.cache_data
def load_data():
    data = yf.download("BTC-USD", start="2020-01-01")
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data['Tomorrow'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
    return data

data = load_data()

# Hiển thị dữ liệu
if st.checkbox("Hiển thị dữ liệu"):
    st.dataframe(data.tail())

# Chọn features và target
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Kết quả dự đoán")
st.write(f"Độ chính xác: {accuracy * 100:.2f}%")

# Dự đoán xu hướng hôm nay
latest_data = data[features].iloc[-1:]
prediction_today = model.predict(latest_data)[0]

if prediction_today == 1:
    st.success("Dự đoán: Giá BTC sẽ TĂNG vào ngày mai.")
else:
    st.error("Dự đoán: Giá BTC sẽ GIẢM vào ngày mai.")
