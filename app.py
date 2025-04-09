import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="AI Dự đoán BTC", layout="centered")
st.title("AI Dự đoán xu hướng Bitcoin")

# --- Load BTC Data ---
@st.cache_data

def load_data():
    df = yf.download("BTC-USD", start="2021-01-01")
    df = df[['Close']].copy()
    df['Tomorrow Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df['Target'] = (df['Tomorrow Close'] > df['Close']).astype(int)
    return df

df = load_data()

st.subheader("Biểu đồ giá BTC")
st.line_chart(df['Close'])

# --- Train model ---
st.subheader("Huấn luyện mô hình")
df['SMA_5'] = df['Close'].rolling(5).mean()
df['SMA_10'] = df['Close'].rolling(10).mean()
df.dropna(inplace=True)

X = df[['Close', 'SMA_5', 'SMA_10']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
st.code(classification_report(y_test, y_pred))

# --- Dự đoán hôm nay ---
st.subheader("Dự đoán xu hướng hôm nay")
latest_data = df.iloc[-1][['Close', 'SMA_5', 'SMA_10']].values.reshape(1, -1)
prediction = model.predict(latest_data)[0]
pred_label = "TĂNG" if prediction == 1 else "GIẢM"

st.metric("Dự đoán xu hướng ngày mai", pred_label)

st.caption("Dữ liệu từ Yahoo Finance. Mô hình: Random Forest.")
