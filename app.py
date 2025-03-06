import streamlit as st
import whisper
import tempfile
import os

import subprocess

# Coba install ffmpeg jika tidak tersedia
try:
    subprocess.run(["ffmpeg", "-version"], check=True)
except FileNotFoundError:
    print("Installing ffmpeg...")
    subprocess.run(["pip", "install", "imageio[ffmpeg]", "ffmpeg-python"])

# Tambahkan lokasi ffmpeg ke PATH (jika perlu)
os.environ["PATH"] += os.pathsep + os.getcwd()


# Load Whisper model (gunakan caching agar tidak reload setiap kali)
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

st.title("🎤 Voice Recorder with Transcriptionn")

# Merekam suara langsung
audio_value = st.audio_input("Record a voice message")

if audio_value:
    st.audio(audio_value, format="audio/wav")

    # Simpan audio ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_value.getvalue())
        temp_audio_path = temp_audio.name

    # Menampilkan animasi loading saat transkripsi berlangsung
    with st.spinner("⏳ Transcribing... Please wait..."):
        result = model.transcribe(temp_audio_path)

    st.success("✅ Transcription Complete!")
    st.text_area("📝 Transcribed Text:", result["text"])

    # Hapus file sementara
    os.remove(temp_audio_path)
