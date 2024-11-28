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
from sentence_transformers import SentenceTransformer
from pathlib import Path
from llama_index.core.node_parser import SimpleNodeParser
from sentence_transformers import SentenceTransformer
from typing import Any, List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter
from transformers import AutoTokenizer
from qdrant_client.http import exceptions as qdrant_exceptions

# switch page
from streamlit_extras.switch_page_button import switch_page


# Styling
with open( "./style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

if 'page' not in st.session_state:
    st.session_state.page = 'New_Chat.py'  # Halaman awal

# File to store chat history
file_name = __file__

# Menentukan path untuk file riwayat chat

# Mengganti ekstensi dari .py menjadi .json
newFile = file_name.replace('.py', '.json')
newFileName = Path(newFile).name
CHAT_HISTORY_FILE = "./histories/" + newFileName #JADIKAN FILENAME

# Functions to handle chat history persistence
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(messages, file)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []  # Return empty list if no history is found

class NomicOllamaEmbedding(OllamaEmbedding):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(base_url="http://127.0.0.1:11434", model_name="nomic-embed-text")
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding("search_query: " + query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self.aget_general_text_embedding("search_query: " + query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding("search_document: " + text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return await self.aget_general_text_embedding("search_document: " +text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding("search_document: " + text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return await asyncio.gather(
            *[self.aget_general_text_embedding("search_document: " + text) for text in texts]
        )

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        result = self._client.embeddings(
            model=self.model_name, prompt=texts, options=self.ollama_additional_kwargs
        )
        return result["embedding"]

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embeddings(
            model=self.model_name, prompt=prompt, options=self.ollama_additional_kwargs
        )
        return result["embedding"]
    

class Chatbot:
    def __init__(self, llm="llama3.2:latest", embedding_model="", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = NomicOllamaEmbedding()
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
ATAUPUN FILE PATH SECARA EXPLICIT."""
        return Settings

    @st.cache_resource(show_spinner=True)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Sedang memuat, sabar yaa."):
            reader = SimpleDirectoryReader(input_dir="./datum_RAG", recursive=True)
            documents = list(reader.load_data())
            
            # Initialize the splitter with chunk size and overlap
            tokenizerModel = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
            token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64, backup_separators=["\n\n"], tokenizer=tokenizerModel)
            split_documents = [token_splitter.split_text(doc.text) for doc in documents]
            
            # Flatten the list of chunks into individual segments
            all_segments = []
            for chunks in split_documents:
                all_segments.extend(chunks)
            
            # Load Hugging Face tokenizer
            
            # Generate embeddings for each segment
            embedder = NomicOllamaEmbedding()
            all_embeddings = Settings.embed_model._get_text_embeddings(all_segments)
            
        # Set up Qdrant collection and client if no vector store is provided
        if vector_store is None:
            client = QdrantClient(
                url=st.secrets["qdrant"]["connection_url"], 
                api_key=st.secrets["qdrant"]["api_key"],
            )
            vector_store = QdrantVectorStore(client=client, collection_name="A3_Split2")
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
            retriever=index.as_retriever(verbose=True),
            llm=Settings.llm
        )

# Main Program
st.title("Splitter & Tokenizer Nomic")

# Initialize chat history or load from file
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()  # Load chat history from file on page load

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])
    if(message["role"] == "assistant"):
        with st.chat_message("assistant", avatar="./chatbot.png"):
            st.markdown(message["content"])
    if(message["role"] == "user"):
        with st.chat_message("user", avatar="./person.jpg"):
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
    with st.chat_message("user", avatar="./person.jpg"):
        st.markdown(prompt)
    add_message("user", prompt)  # Save user message

    # Get AI response
    with st.chat_message("assistant", avatar="./chatbot.png"):
        response = chatbot.chat_engine.chat(prompt)
        st.markdown(response.response)
    add_message("assistant", response.response)  # Save assistant response

st.sidebar.header("Control Point")
redirect_to_home = False
if st.sidebar.button("Delete"):
    namaFileOriginal = __file__
    pythonFile = Path(namaFileOriginal).name
    jsonFile = pythonFile.replace('.py', '.json')
    os.remove("./histories/"+jsonFile)
    os.remove(namaFileOriginal)
    switch_page("main_page")