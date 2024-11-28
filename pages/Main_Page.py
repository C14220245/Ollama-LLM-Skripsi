import streamlit as st
import os
import shutil
import json
from ollama import Client
import time
from typing import Final
from streamlit_js_eval import streamlit_js_eval
st.title("ChatBot Creator")

# Styling
with open( "./style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
#Css untuk button di sidebar
st.markdown("""
<style>
.sidebar .stButton > button {
    background-color: #4CAF50;  /* Ganti dengan warna yang Anda inginkan */
    color: white;  /* Warna teks */
    font-size: 16px;
    border-radius: 5px;
}

.sidebar .stButton > button:hover {
    background-color: #45a049;  /* Warna saat hover */
}
</style>
""", unsafe_allow_html=True)

# # Inisialisasi counter global
# if "refresh" not in st.session_state:
#     st.session_state.refresh = 0

# if st.session_state.refresh == 0:
#     streamlit_js_eval(js_expressions="parent.window.location.reload()")

# st.session_state.refresh += 1
    

# Inisialisasi counter global
if "count" not in st.session_state:
    st.session_state.count = 0

# Inisialisasi nama file JSON untuk menyimpan chat history
if "json_filename" not in st.session_state:
    st.session_state.json_filename = None

# Nama page untuk sidebar
PAGES_DIR = "pages"

# Set Ollama Client
client = Client(host="http://127.0.0.1:11434")


# handles stream response back from LLM
def stream_parser(stream):
    for chunk in stream:
        yield chunk['message']['content']

# Set a default model
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "llama3.1:latest"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
if "count" not in st.session_state:
    for message in st.session_state.messages:
        # with st.chat_message(message["role"]):
        #     st.markdown(message["content"])
        if(message["role"] == "assistant"):
            with st.chat_message("assistant", avatar="./chatbot.png"):
                st.markdown(message["content"])
        if(message["role"] == "user"):
            with st.chat_message("user", avatar="./person.jpg"):
                st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Jika ini adalah prompt pertama, buat file JSON untuk pertama kali
    if st.session_state.count == 0:
        
        # Nama file berdasarkan input pertama
        filenameForJSON = prompt.replace(" ", "_") + ".json"
        file_path = os.path.join("histories", filenameForJSON)

        # Simpan nama file JSON ke session state
        st.session_state.json_filename = file_path

        # Buat file JSON baru
        with open(file_path, 'w') as json_file:
            json.dump([], json_file, indent=4)  # Inisialisasi file dengan list kosong

        # Jika perlu membuat file halaman
        filename = prompt.replace(" ", "_") + ".py"
        source_file = "templates/template_page.py"  # File template yang ingin diduplikasi
        destination_file = os.path.join(PAGES_DIR, filename)  # Path file tujuan

        # try:
            # Menyalin file template ke folder pages dengan nama baru
        shutil.copy(source_file, destination_file)

            # Beri notifikasi bahwa file berhasil dibuat
        #     st.success(f"File '{filename}' berhasil dibuat di folder 'pages'.")
        # except Exception as e:
        #     st.error(f"Terjadi kesalahan saat menyalin file: {e}")
        # else:
        #     st.warning("Mohon masukkan judul untuk halaman.")

    # Tambah hitungan prompt
    st.session_state.count += 1

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user", avatar="./person.jpg"):
        st.markdown (prompt)

    # Tampilkan respons asisten
    with st.chat_message("assistant", avatar="./chatbot.png"):
        stream = client.chat(
            model=st.session_state["ollama_model"],
            messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            stream=True,
        )
        response = st.write_stream(stream_parser(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Buka file JSON dan tambahkan pesan baru
    if st.session_state.json_filename:
        with open(st.session_state.json_filename, 'r+') as json_file:
            try:
                chat_history = json.load(json_file)  # Membaca data yang ada

                # Tambahkan input pengguna dan respons ke dalam chat history
                chat_history.append({"role": "user", "content": prompt})
                chat_history.append({"role": "assistant", "content": response})

                # Kembali ke awal file dan simpan update chat history
                json_file.seek(0)
                json.dump(chat_history, json_file, indent=4)
            except json.JSONDecodeError:
                st.error("Terjadi kesalahan dalam membaca file JSON.")
            except Exception as e:
                st.error(f"Error: {e}")