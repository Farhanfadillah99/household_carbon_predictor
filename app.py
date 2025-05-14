import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from carbonpredictor import train_model

# Config
st.set_page_config(page_title="AI Green Tech")
st.title("AI Green Tech – Carbon Emission Predictor")
st.caption("Prediksi emisi karbon rumah tangga + visualisasi & tips")

# Load model
model_path = 'model/carbon_predictor.pkl'
if not os.path.exists(model_path):
    st.warning("Model belum ditemukan. Melatih model...")
    train_model()

model = joblib.load(model_path)

# Dummy average data (untuk visualisasi)
avg_emission = 150  # dummy data rata-rata karbon bulanan rumah tangga

# Input user
electricity = st.number_input("Konsumsi Listrik (kWh/bulan)", min_value=0, value=250)
gas = st.number_input("Konsumsi Gas (m³/bulan)", min_value=0, value=30)
transport = st.number_input("Jarak Transportasi (km/bulan)", min_value=0, value=200)

if st.button("Hitung Emisi Karbon"):
    input_data = np.array([[electricity, gas, transport]])
    prediction = model.predict(input_data)
    carbon = prediction[0]

    st.success(f"Estimasi Emisi: **{carbon:.2f} kg CO₂/bulan**")

    # Visualisasi
    st.subheader("Perbandingan Emisi")
    df_plot = pd.DataFrame({
        'Kategori': ['Rata-rata Nasional', 'Kamu'],
        'Emisi (kg CO₂/bulan)': [avg_emission, carbon]
    })

    fig, ax = plt.subplots()
    bars = ax.bar(df_plot['Kategori'], df_plot['Emisi (kg CO₂/bulan)'], color=['gray', 'green'])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}', ha='center', va='bottom')
    st.pyplot(fig)

    # Tips
    st.subheader("Rekomendasi Ramah Lingkungan")
    tips = []

    if electricity > 300:
        tips.append("Kurangi pemakaian listrik: Gunakan lampu LED & matikan alat saat tidak digunakan.")
    if gas > 40:
        tips.append("Pertimbangkan isolasi rumah yang lebih baik atau gunakan kompor induksi.")
    if transport > 250:
        tips.append("Gunakan transportasi umum, sepeda, atau carpool untuk mengurangi jejak karbon.")
    
    if not tips:
        st.info("Aktivitasmu sudah cukup efisien. Pertahankan gaya hidup ramah lingkungan!")
    else:
        for tip in tips:
            st.write(tip)

# Footer
st.markdown("---")
st.markdown("Made by Farhan Fadillah")
