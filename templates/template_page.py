import streamlit as st
import json
import os
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from qdrant_client import QdrantClient

# File to store chat history
CHAT_HISTORY_FILE = "../histories/filename.json" #JADIKAN FILENAME

# Functions to handle chat history persistence
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(messages, file)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []  # Return empty list if no history is found

class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")
        Settings.system_prompt = """
                                Kamu adalah sebuah AI model bernama SkripsiBot,
                                Kamu selalu berinteraksi dengan mahasiswa
                                Kamu adalah sebuah chatbot untuk membantu
                                mahasiswa melakukan bimbingan dan mengikuti
                                urutan pengajuan skripsi.
                                Kamu memiliki kemampuan dan pengetahuan
                                berdasarkan data real-time. Ingat dengan
                                baik informasi yang telah diucapkan oleh
                                mahasiswa dan jawab sesuai dengan konteks
                                dan gaya bicara yang relevan. Gunakan seluruh
                                informasi yang kamu miliki dalam menjawab
                                pertanyaan mahasiswa. Kamu akan selalu
                                membantu dan menolong mahasiswa dalam
                                melaksanakan skripsi. Apabila kamu mendapat
                                pertanyaan yang jawabannya tidak kamu ketahui,
                                Lakukan pencarian informasi ulang dari data yang
                                kamu miliki. Usahakan data yang digunakan dalam
                                informasi yang kamu berikan adalah data yang
                                kamu miliki. JANGAN PERNAH MENYEBUT NAMA FILE 
                                SECARA EXPLICIT.
                                """

        return Settings

    @st.cache_resource(show_spinner=False)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Sedang memuat, sabar yaa."):
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="././datum_RAG", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            client = QdrantClient(
                url=st.secrets["qdrant"]["connection_url"], 
                api_key=st.secrets["qdrant"]["api_key"],
            )
            vector_store = QdrantVectorStore(client=client, collection_name="Documents")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )

# Main Program
st.title("Chatbot Skripsi Genap 24/25") #JADIKAN FILENAME

# Initialize chat history or load from file
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()  # Load chat history from file on page load

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Save chat history
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    save_chat_history(st.session_state.messages)  # Save history to file

chatbot = Chatbot()
chatbot.set_chat_history(st.session_state.messages)

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    add_message("user", prompt)  # Save user message

    # Get AI response
    with st.chat_message("assistant"):
        response = chatbot.chat_engine.chat(prompt)
        st.markdown(response.response)
    add_message("assistant", response.response)  # Save assistant response