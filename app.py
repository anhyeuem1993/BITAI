import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Dự đoán BTC", layout="wide")

@st.cache_data
def load_data():
    today = datetime.today()
    start = today - timedelta(days=60)
    df = yf.download('BTC-USD', start=start.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df['Tomorrow Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow Close'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

data = load_data()

st.subheader("Dữ liệu giá Bitcoin")
st.dataframe(data.tail())

# Train model
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Dự đoán hôm nay
latest_data = data[features].iloc[-1:]
prediction = model.predict(latest_data)[0]
prob = model.predict_proba(latest_data)[0]

st.subheader("Dự đoán xu hướng hôm nay:")
if prediction == 1:
    st.success(f"AI dự đoán: Tăng (xác suất {prob[1]*100:.2f}%)")
else:
    st.error(f"AI dự đoán: Giảm (xác suất {prob[0]*100:.2f}%)")
