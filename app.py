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
