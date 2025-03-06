import os
import opensmile
import pandas as pd
import numpy as np
import joblib
import tempfile
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load model dan parameter preprocessing
rf_model = joblib.load("trained_rfmodel_RT95.pkl")
scaler = joblib.load("scaler_RT95.pkl")
selector = joblib.load("rfe_feature_selector_RT95.pkl")
pca = joblib.load("pca_model_RT95.pkl")

print("Model dan preprocessing tools telah dimuat kembali.")

# Inisialisasi OpenSMILE untuk ekstraksi fitur
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPS,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def predict_audio(file_path):
    try:
        with tqdm(total=100, desc="Processing Audio", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            feature_vector = smile.process_file(file_path)
            features = feature_vector.values.flatten().reshape(1, -1)
            pbar.update(30)

            X_scaled = scaler.transform(features)
            pbar.update(30)

            X_selected = selector.transform(X_scaled)
            X_pca = pca.transform(X_selected)
            pbar.update(30)

            probabilities = rf_model.predict_proba(X_pca)[0]
            y_pred = np.argmax(probabilities)
            pbar.update(10)

            labels = ["Healthy", "Depressed"]
            result_label = labels[y_pred]
            confidence = probabilities[y_pred] * 100

            return result_label, confidence
    except Exception as e:
        return f"Error dalam prediksi: {e}", None

st.title("Aplikasi Deteksi Depresi")

st.write("## Rekam Suara")
audio_value = st.audio_input("Record a voice message")

audio_path = None

if audio_value:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_value.read())
        audio_path = tmp_audio.name
    st.audio(audio_value)

st.write("## Atau Unggah File Audio")
uploaded_file = st.file_uploader("Unggah file audio (.wav) untuk analisis:", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

if audio_path:
    st.write("\n**Menganalisis audio...**")
    prediction_result, confidence = predict_audio(audio_path)

    if confidence is not None:
        st.write(f"**Hasil Prediksi:** {prediction_result} ({confidence:.2f}% confidence)")
