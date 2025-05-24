import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# ตั้งค่าฟอนต์
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# โหลดข้อมูล
df = pd.read_excel("D:/NDR_project/PLTR_dataset.xlsx", sheet_name="PLTR", skiprows=1)
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol", "Change %", "Set index"]

# ทำความสะอาดข้อมูล
df = df[~df["Date"].isna() & ~df["Date"].astype(str).str.contains("Date")]

# แปลงวันที่จาก MM/DD/YY เป็น datetime
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")

# ลบแถวที่มีวันที่แปลงไม่ได้ (NaT)
df = df.dropna()

# กรองข้อมูลถึงวันที่ปัจจุบัน (25 พฤษภาคม 2568)
current_date = pd.to_datetime("2025-05-25")
df = df[df["Date"] <= current_date]

# เรียงข้อมูลตามวันที่
df_sorted = df.sort_values("Date")

# ใช้ df_sorted สำหรับการแสดงผล (แปลงวันที่เป็น string เพื่อแสดงในตาราง)
df_sorted_display = df_sorted.copy()
df_sorted_display["Date"] = df_sorted_display["Date"].dt.strftime('%Y-%m-%d')

# ตั้งชื่อหน้า
st.title("📈 วิเคราะห์ราคาหุ้น PLTR")

# ส่วนที่ 1: กราฟแนวโน้มราคาหุ้น
st.subheader("📉 แนวโน้มราคาหุ้น PLTR")
X = df_sorted["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df_sorted["Price"].values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_sorted["Date"], y, label="Actual Closing Price")
ax.plot(df_sorted["Date"], trend, label="Trend (Linear Regression)", linestyle="--", color="red")
ax.set_title("PLTR Closing Price Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (Baht)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ส่วนที่ 2: ข้อมูลเบื้องต้น (แสดง 10 แถวล่าสุด)
st.subheader("🧾 ข้อมูลเบื้องต้น")
st.dataframe(
    df_sorted_display[["Date", "Price", "Open", "High", "Low", "Set index"]].tail(10).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)
