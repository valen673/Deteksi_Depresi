import streamlit as st
import json
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Inisialisasi OpenAI Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Inisialisasi memori percakapan
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Template Prompt untuk Skrining Depresi
template = PromptTemplate.from_template("""
Anda adalah chatbot psikologi yang melakukan skrining depresi.
Gunakan pertanyaan berikut untuk memandu percakapan:
1. Bagaimana perasaan Anda hari ini?
2. Apakah Anda merasa sedih dalam 2 minggu terakhir?
3. Apakah Anda mengalami gangguan tidur?

Ingat percakapan sebelumnya dan sesuaikan pertanyaan Anda.

Percakapan sebelumnya:
{history}

User: {input}
Chatbot:
""")

# UI Streamlit
st.title("Chatbot Skrining Depresi")
st.write("Halo, saya di sini untuk membantu Anda melakukan skrining depresi. Bagaimana perasaan Anda hari ini?")

# Input pengguna
user_input = st.text_input("Anda:", "")

if st.button("Kirim") and user_input:
    messages = st.session_state.memory.chat_memory.messages + [{"role": "user", "content": user_input}]
    
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Chatbot Skrining Depresi",
        },
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=messages
    )
    response = completion.choices[0].message.content
    
    st.session_state.memory.save_context({"input": user_input}, {"output": response})
    
    # Tampilkan chat history
    st.write("**Chatbot:**", response)

# Tombol Download Hasil Skrining
data = {
    "user_id": "123",
    "responses": st.session_state.memory.buffer
}
json_data = json.dumps(data, indent=4)
st.download_button(
    label="Download Hasil Skrining",
    data=json_data,
    file_name="chat_history.json",
    mime="application/json"
)
