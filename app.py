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

st.markdown(
    """
    <div style="border: 2px solid #000000; padding: 15px; border-radius: 10px; background-color: #000000; text-align: justify; color: #ffffff;">
    <strong>Silahkan baca cerita berikut untuk mendapatkan hasil prediksi depresi:</strong><br><br>
    Angin utara dan matahari berdebat tentang siapa di antara mereka yang lebih kuat, ketika tiba-tiba muncul seorang pengembara yang mengenakan mantel tebal. Mereka pun sepakat bahwa di antara mereka, siapa pun yang pertama kali berhasil membuat pengembara melepas mantelnya akan dianggap sebagai yang terkuat.<br><br>
    Angin utara mulai bertiup sekuat tenaga, tetapi semakin kencang ia bertiup, semakin erat pengembara membungkus dirinya dengan mantelnya. Akhirnya, angin utara memutuskan untuk menyerah.<br><br>
    Lalu, matahari mulai bersinar terang, dan pengembara segera melepaskan mantelnya. Saat itulah angin utara harus mengakui bahwa matahari adalah yang terkuat di antara mereka.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Unggah file audio (.wav) untuk analisis:", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filepath = tmp_file.name

    st.write("\n**Menganalisis audio...**")
    prediction_result, confidence = predict_audio(tmp_filepath)

    if confidence is not None:
        st.session_state.prediction_result = prediction_result
        st.session_state.confidence = confidence
        st.session_state.show_questionnaire = True

if st.session_state.get("show_questionnaire", False):
    st.write("## Kuisioner PHQ-9")
    st.write("Dalam 2 minggu terakhir, seberapa sering Anda terganggu oleh masalah-masalah berikut? (Gunakan dropdown untuk memilih)")

    questions = [
        "Kurang berminat atau bergairah dalam melakukan apapun",
        "Merasa murung, sedih, atau putus asa",
        "Sulit tidur/mudah terbangun, atau terlalu banyak tidur",
        "Merasa lelah atau kurang bertenaga",
        "Kurang nafsu makan atau terlalu banyak makan",
        "Kurang percaya diri — atau merasa bahwa Anda adalah orang yang gagal atau telah mengecewakan diri sendiri atau keluarga",
        "Sulit berkonsentrasi pada sesuatu, misalnya membaca koran atau menonton televisi",
        "Bergerak atau berbicara sangat lambat sehingga orang lain memperhatikannya. Atau sebaliknya; merasa resah atau gelisah sehingga Anda lebih sering bergerak dari biasanya.",
        "Merasa lebih baik mati atau ingin melukai diri sendiri dengan cara apapun."
    ]

    options = {"Tidak Pernah": 0, "Beberapa hari": 1, "Lebih dari separuh waktu": 2, "Hampir setiap hari": 3}

    for i, question in enumerate(questions):
        if f"q{i}" not in st.session_state:
            st.session_state[f"q{i}"] = "Tidak Pernah"
        st.session_state[f"q{i}"] = st.selectbox(f"{i+1}. {question}", list(options.keys()), index=list(options.keys()).index(st.session_state[f"q{i}"]))

    if st.button("Selesai"):
        total_score = sum(options[st.session_state[f"q{i}"]] for i in range(len(questions)))

        if total_score <= 4:
            phq_result = "Tidak ada gejala depresi"
        elif total_score <= 9:
            phq_result = "Gejala depresi ringan"
        elif total_score <= 14:
            phq_result = "Depresi ringan"
        elif total_score <= 19:
            phq_result = "Depresi sedang"
        else:
            phq_result = "Depresi berat"

        st.session_state.phq_result = phq_result
        st.session_state.total_score = total_score
        st.session_state.show_report = True

if st.session_state.get("show_report", False):
    st.markdown(
        f"""
        <div style="border: 2px solid #000000; padding: 15px; border-radius: 10px; background-color: #000000; text-align: justify; color: #ffffff;">
        <strong>Laporan Skrining Depresi</strong><br><br>
        - **Hasil Prediksi Audio**: {st.session_state.prediction_result} ({st.session_state.confidence:.2f}% confidence)<br>
        - **Hasil Kuisioner PHQ-9**: {st.session_state.phq_result} (Skor: {st.session_state.total_score})
        </div>
        """,
        unsafe_allow_html=True
    )
