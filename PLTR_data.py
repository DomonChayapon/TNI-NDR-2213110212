import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ตั้งค่าฟอนต์
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ฟังก์ชันคำนวณ MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()  # EMA 12 วัน
    exp2 = data.ewm(span=slow, adjust=False).mean()  # EMA 26 วัน
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()  # EMA 9 วันของ MACD
    histogram = macd - signal_line
    return macd, signal_line, histogram

# ฟังก์ชันคำนวณ RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ฟังก์ชันคำนวณ Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()  # Simple Moving Average
    std = data.rolling(window=window).std()  # Standard Deviation
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

# โหลดข้อมูล
df = pd.read_excel("PLTR_dataset.xlsx", sheet_name="PLTR", skiprows=1)
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol", "Change %", "Set index"]

# ทำความสะอาดข้อมูล
df = df[~df["Date"].isna() & ~df["Date"].astype(str).str.contains("Date")]

# แปลงวันที่จาก MM/DD/YY เป็น datetime
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")

# ลบแถวที่มีวันที่แปลงไม่ได้ (NaT)
df = df.dropna()

# กรองข้อมูลถึงวันที่ปัจจุบัน (25 พฤษภาคม 2568)
current_date = pd.to_datetime("2025-05-25")
df = df[df["Date"] <= current_date]

# เรียงข้อมูลตามวันที่ (สำหรับกราฟ)
df_sorted = df.sort_values("Date")

# คำนวณ MACD, RSI, และ Bollinger Bands
macd, signal_line, histogram = calculate_macd(df_sorted["Price"])
rsi = calculate_rsi(df_sorted["Price"])
sma, upper_band, lower_band = calculate_bollinger_bands(df_sorted["Price"])

# เพิ่มผลลัพธ์ลงใน DataFrame
df_sorted["MACD"] = macd
df_sorted["Signal"] = signal_line
df_sorted["Histogram"] = histogram
df_sorted["RSI"] = rsi
df_sorted["SMA"] = sma
df_sorted["Upper Band"] = upper_band
df_sorted["Lower Band"] = lower_band

# ตั้งชื่อหน้าและเพิ่มไอคอน
col1, col2 = st.columns([1, 7])
with col1:
    st.image("pltr_icon.png", width=80)  # ใช้ขนาดแนะนำ width=80
with col2:
    st.title("Palantir Technologies Inc")
st.markdown("<small>NASDAQ: PLTR</small>", unsafe_allow_html=True)

# ข้อมูลบริษัทใน 3 บรรทัด ขนาดตัวอักษรเล็กลง
st.markdown(
    "<small>Palantir Technologies Inc. is a U.S.-based software company founded in 2003, specializing in big data analytics.<br>"
    "Headquartered in Denver, Colorado, it offers platforms: Gotham (defense), Foundry (enterprise), Apollo (CI/CD), and AIP.<br>"
    "Used by government agencies and companies like Morgan Stanley, Airbus, and Fiat Chrysler.</small>"
    , unsafe_allow_html=True
)

# ส่วนที่ 2: กราฟแนวโน้มราคาหุ้น
days_option = st.selectbox("Select time period (day)", ["All", 15, 30, 60, 90])
time_period = "All" if days_option == "All" else f"{days_option} days"
st.subheader(f"Price chart: {time_period}")

# กรองข้อมูลตามจำนวนวัน
if days_option == "All":
    df_filtered = df_sorted
else:
    df_filtered = df_sorted[df_sorted["Date"] >= (current_date - pd.Timedelta(days=days_option))]

# คำนวณ MACD, RSI, และ Bollinger Bands สำหรับข้อมูลที่กรองแล้ว
macd_filtered, signal_line_filtered, histogram_filtered = calculate_macd(df_filtered["Price"])
rsi_filtered = calculate_rsi(df_filtered["Price"])
sma_filtered, upper_band_filtered, lower_band_filtered = calculate_bollinger_bands(df_filtered["Price"])

# เพิ่มผลลัพธ์ลงใน DataFrame ที่กรองแล้ว
df_filtered = df_filtered.copy()
df_filtered["MACD"] = macd_filtered
df_filtered["Signal"] = signal_line_filtered
df_filtered["Histogram"] = histogram_filtered
df_filtered["RSI"] = rsi_filtered
df_filtered["SMA"] = sma_filtered
df_filtered["Upper Band"] = upper_band_filtered
df_filtered["Lower Band"] = lower_band_filtered

# Dropdown สำหรับเลือกประเภทกราฟ
chart_option = st.selectbox("Select graph type", ["Trend", "Indicator"])

# ตัวแปรสำหรับควบคุมการแสดงกราฟ
show_macd = False
show_rsi = False
show_bollinger = False

# ถ้าเลือก Indicator ให้แสดง Checkbox
if chart_option == "Indicator":
    show_macd = st.checkbox("Show MACD", value=False)
    show_rsi = st.checkbox("Show RSI", value=False)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=False)

# เตรียมข้อมูลสำหรับกราฟ
if chart_option == "Indicator":
    # จำนวนแถวของ subplot (เริ่มต้นที่ 1 สำหรับ Candlestick)
    rows = 1
    if show_macd:
        rows += 1
    if show_rsi:
        rows += 1
    if show_bollinger:
        rows += 1

    # สร้าง subplot
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=["Candlestick", "MACD" if show_macd else "", "RSI" if show_rsi else "", "Bollinger Bands" if show_bollinger else ""])

    # เพิ่มกราฟแท่งเทียนในแถวแรก
    fig.add_trace(
        go.Candlestick(
            x=df_filtered["Date"],
            open=df_filtered["Open"],
            high=df_filtered["High"],
            low=df_filtered["Low"],
            close=df_filtered["Price"],
            name="PLTR"
        ),
        row=1, col=1
    )

    # เพิ่มกราฟ MACD ถ้าติ๊ก
    current_row = 2
    if show_macd:
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["MACD"], mode="lines", name="MACD", line=dict(color="blue")),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["Signal"], mode="lines", name="Signal", line=dict(color="orange")),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(x=df_filtered["Date"], y=df_filtered["Histogram"], name="Histogram", marker_color="grey"),
            row=current_row, col=1
        )
        current_row += 1

    # เพิ่มกราฟ RSI ถ้าติ๊ก
    if show_rsi:
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["RSI"], mode="lines", name="RSI", line=dict(color="purple")),
            row=current_row, col=1
        )
        # เพิ่มเส้น 70 และ 30 (ระดับ Overbought/Oversold)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1

    # เพิ่มกราฟ Bollinger Bands ถ้าติ๊ก
    if show_bollinger:
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["Upper Band"], mode="lines", name="Upper Band", line=dict(color="red", dash="dash")),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["SMA"], mode="lines", name="SMA", line=dict(color="black")),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_filtered["Date"], y=df_filtered["Lower Band"], mode="lines", name="Lower Band", line=dict(color="green", dash="dash")),
            row=current_row, col=1
        )

    # ปรับแต่งกราฟ
    fig.update_layout(
        title="PLTR Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (Baht)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="zoom",
        template="plotly_white",
        showlegend=True,
        height=300 * rows  # ปรับความสูงตามจำนวน subplot
    )

    # ปรับแต่งแกน y สำหรับ subplot
    fig.update_yaxes(title_text="Price (Baht)", row=1, col=1)
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=2, col=1)
    if show_rsi:
        fig.update_yaxes(title_text="RSI", row=3 if show_macd else 2, col=1, range=[0, 100])
    if show_bollinger:
        fig.update_yaxes(title_text="Bollinger Bands", row=4 if show_macd and show_rsi else 3 if show_macd or show_rsi else 2, col=1)

    # แสดงกราฟใน Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif chart_option == "Trend":
    # กราฟเส้นตามเดิม
    X = df_filtered["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = df_filtered["Price"].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered["Date"], y, label="Actual Closing Price")
    ax.plot(df_filtered["Date"], trend, label="Trend (Linear Regression)", linestyle="--", color="red")
    ax.set_title("PLTR Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (Baht)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# ส่วนที่ 2: ข้อมูลเบื้องต้น (แสดงตามจำนวนวันที่เลือก)
st.subheader(f"Historical price: {time_period}")
# คัดลอก df_filtered เพื่อปรับแต่งการแสดงผล
df_display = df_filtered.copy()
# ลบเวลาออก แสดงเฉพาะวันที่
df_display["Date"] = df_display["Date"].dt.strftime('%Y-%m-%d')
# เรียงวันที่จากล่าสุดไปเก่าสุด
df_display = df_display.sort_values("Date", ascending=False)
# กำหนดจำนวนแถวตามวันที่เลือก
if days_option == "All":
    rows_to_show = len(df_filtered)
else:
    rows_to_show = min(days_option, len(df_filtered))
st.dataframe(
    df_display[["Date", "Price", "Open", "High", "Low", "Set index"]].head(rows_to_show).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)