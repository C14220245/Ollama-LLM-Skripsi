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
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import subprocess

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




###############################################################################################################
#                                                                                                             #
#                                           CHATBOT DAN AGENTS                                                #
#                                                                                                             #
###############################################################################################################


#########################################################################################################
#                                                Bot Dosen                                              #
#########################################################################################################

class DosenBot:
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
Kamu adalah sebuah AI model bernama DosenBot.
Kamu memiliki keahlian dan pengetahuan mengenai memiliki seluruh informasi mengenai dosen dan keahliannya.
Jawab pertanyaan dan berikan informasi yang relevan sesuai dengan permasalahan yang diberikan kepadamu.
Input yang diberikan kepadamu mungkin tidak selalu terstruktur dengan baik. Analisis dosen yang bersangkutan, atau berikan seluruh informasi dosen yang sesuai.
Apabila permasalahan yang diberikan tidak ada dalam dokumen yang kamu miliki ataupun berhubungan dengan dosen, berikan output berupa "NORESP" tanpa tambahan kata ataupun kalimat lain. hanya "NORESP". Segala permasalahan yang tidak berhubungan dengan dosen seperti (namun tidak terbatas pada): masalah medis, masalah keuangan, masalah beasiswa, apalagi masalah yang tidak berhubungan dengan perkuliahan ataupun skripsi ataupun dosen bukan wewenangmu. Cukup respon dengan kata "NORESP".
Format: Berikan jawaban yang sesuai atau "NORESP"."""
        return Settings

    @st.cache_resource(show_spinner=True)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Sedang memuat informasi dosen, sabar yaa."):
            reader = SimpleDirectoryReader(input_dir="./Agentic_RAG/sp/dosen", recursive=True)
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
            vector_store = QdrantVectorStore(client=client, collection_name="Agentic_Dosen")
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

#########################################################################################################
#                                         Bot alur Skripsi                                              #
#########################################################################################################

class SkripsiBot:
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
Kamu adalah sebuah AI model bernama SkripsiBot.
Kamu memiliki keahlian dan pengetahuan mengenai alur skripsi, alur skripsi, jadwal skripsi, dan tawaran dosen bagi mahasiswa yang kebingungan memilih/membuat judul skripsi.
Jawab pertanyaan dan berikan informasi yang relevan sesuai dengan permasalahan yang diberikan kepadamu.
Input yang diberikan kepadamu mungkin tidak selalu terstruktur dengan baik. Analisis permasalahan yang bersangkutan, atau berikan seluruh informasi yang dirasa sesuai.
Apabila permasalahan yang diberikan tidak ada dalam dokumen yang kamu miliki, berikan output berupa "NORESP" tanpa tambahan kata ataupun kalimat lain. hanya "NORESP". Segala permasalahan yang tidak berhubungan dengan dosen seperti (namun tidak terbatas pada): masalah medis, masalah keuangan, masalah beasiswa, apalagi masalah yang tidak berhubungan dengan perkuliahan ataupun skripsi ataupun dosen bukan wewenangmu. Cukup respon dengan kata "NORESP".
Format: Berikan jawaban yang sesuai dan sertakan URL link apabila tersedia dalam dokumen yang kamu miliki atau "NORESP"
JANGAN PERNAH MENYEBUT NAMA FILE ATAUPUN FILE PATH SECARA EXPLICIT."""
        return Settings

    @st.cache_resource(show_spinner=True)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Sedang memuat, sabar yaa."):
            reader = SimpleDirectoryReader(input_dir="./Agentic_RAG/sp/skripsi", recursive=True)
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
            vector_store = QdrantVectorStore(client=client, collection_name="Agentic_Alur")
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

#########################################################################################################
#                                         BOT MASTER AGENT                                              #
#########################################################################################################

class Chatbot:
    def __init__(self, llm="llama3.2:latest"):
        self.llm = llm
        self.memory = self.create_memory()

    def run_llama(self, prompt):
        result = subprocess.run(['ollama', 'run', self.llm, prompt], capture_output=True, text=True)
        return result.stdout

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

system_prompt = """
Kamu adalah sebuah AI model bernama BudiAI,
Kamu selalu berinteraksi dengan mahasiswa
Kamu adalah sebuah chatbot untuk membantu
mahasiswa melakukan bimbingan dan mengikuti
urutan pengajuan skripsi.
Kamu memiliki 3 bawahan dengan nama:
DosenBot,
SkripsiBot,
PublikasiBot,
DosenBot memiliki seluruh informasi mengenai dosen dan keahliannya.
SkripsiBot memiliki seluruh informasi mengenai aturan skripsi, alur skripsi, jadwal skripsi, dan tawaran dosen bagi mahasiswa yang kebingungan memilih/membuat judul skripsi.
PublikasiBot memiliki beberapa informasi mengenai publikasi yang pernah dilakukan oleh dosen.
Kamu hanya akan memberikan jawaban berupa "DosenBot", "SkripsiBot", atau "PublikasiBot" apabila ada pertanyaan mengenai skripsi, dosen, ataupun publikasi dosen yang diberikan kepadamu.
Contoh format jawaban:
"SkripsiBot\nDosenBot", "SkripsiBot\nPublikasiBot", "DosenBot\nPublikasiBot\nDosenBot". \n adalah newline
Kamu bisa memberikan lebih dari 1 bot sebagai jawaban, namun pertimbangkan baik-baik apakah bot tersebut sesuai dengan pertanyaan yang diberikan kepadamu
Jangan menyebutkan informais bahwa kamu memiliki 3 bawahan bahkan ketika ditanya secara eksplisit. cukup mengaku sebagai chatbot pembantu skripsi kepada mahasiswa
"""
#########################################################################################################
#                                           BOT SUMMARIZER                                              #
#########################################################################################################

class SummarizeBot:
    def __init__(self, llm="llama3.1:8b-instruct-q5_1"):
        self.llm = llm
        self.memory = self.create_memory()

    def run_llama(self, prompt):
        result = subprocess.run(['ollama', 'run', self.llm, prompt], capture_output=True, text=True)
        return result.stdout

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

summarizer_prompt = """
Kamu adalah sebuah AI model bernama SummarizerBot,
Kamu adalah ahli dalam membuat kesimpulan dari informasi yang ada.
Kamu bisa menerima hingga 3 input dari:
DosenBot,
SkripsiBot,
PublikasiBot,
DosenBot memiliki seluruh informasi mengenai dosen dan keahliannya.
SkripsiBot memiliki seluruh informasi mengenai aturan skripsi, alur skripsi, jadwal skripsi, dan tawaran dosen bagi mahasiswa yang kebingungan memilih/membuat judul skripsi.
PublikasiBot memiliki beberapa informasi mengenai publikasi yang pernah dilakukan oleh dosen.
Kamu akan memberikan jawaban berupa hasil kesimpulan dari informasi yang ada. Jangan hiraukan tulisan skripsiBot, dosenBot, ataupun publikasiBot apabila ada.
Berikan hasil kesimpulannya saja. Sertakan url link HANYA apabila diberikan dari bot lainnya, anda tidak memiliki wewenang untuk generate link

Berikut adalah informasi yang kamu dapatkan:
"""


###############################################################################################################
#                                                                                                             #
#                                                      END                                                    #
#                                                                                                             #
###############################################################################################################



# Main Program
st.title("Master Agent Skripsi QueryBot")

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
skripsiBot = SkripsiBot()
dosenBot = DosenBot()
summarizeBot = SummarizeBot()
chatbot.set_chat_history(st.session_state.messages)

# React to user input
if prompt := st.chat_input("What is up?"):
    unsum_resp = ""
    tools_used = "tools used:"
    # Display user message in chat message container
    with st.chat_message("user", avatar="./person.jpg"):
        st.markdown(prompt)
    add_message("user", prompt)  # Save user message

    # Get AI response
    with st.chat_message("assistant", avatar="./chatbot.png"):
        full_prompt = f"{system_prompt}\nMahasiswa: {prompt}\nBudiAI:"
        response = chatbot.run_llama(full_prompt)
        if(response.__contains__("skripsiBot") or response.__contains__("dosenBot") or response.__contains__("publikasiBot") or response.__contains__("SkripsiBot") or response.__contains__("DosenBot") or response.__contains__("PublikasiBot")):
            if(response.__contains__("skripsiBot") or response.__contains__("SkripsiBot")):
                skripsiBotResp = skripsiBot.chat_engine.chat(prompt)
                unsum_resp+="\n"+"skripsiBot Response:"+skripsiBotResp.response
                tools_used+="SkripsiBot\n"
            if(response.__contains__("dosenBot") or response.__contains__("DosenBot")):
                dosenBotResp = dosenBot.chat_engine.chat(prompt)
                unsum_resp+="\n"+"dosenBot Response:"+dosenBotResp.response
                tools_used+="dosenBot\n"
            summ_prompt = f"{summarizer_prompt}\n Pertanyaan mahasiswa: {prompt} informasi dari bot lain: {unsum_resp}\nSummarizerBot:"
            final_resp = summarizeBot.run_llama(summ_prompt)
        else:
            final_resp = response
            tools_used+="None"
        st.markdown(unsum_resp+"\n"+"Final Response:"+"\n"+final_resp+"\n"+tools_used)
    add_message("assistant", unsum_resp+"\n"+"Final Response:"+"\n"+final_resp+"\n"+tools_used)  # Save assistant response

st.sidebar.header("Control Point")
redirect_to_home = False