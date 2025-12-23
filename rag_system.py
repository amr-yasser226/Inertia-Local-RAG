import os
import shutil
import streamlit as st

# --- LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

# --- OLLAMA IMPORTS (For both LLM and Embeddings) ---
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PERSIST_DIRECTORY = "./chroma_db_qwen"

# ---------------------------------------------------------
# CORE RAG SYSTEM CLASS
# ---------------------------------------------------------
class RAGSystem:
    def __init__(self):
        try:
            # 1. Initialize Embeddings
            self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            
            # 2. Initialize Vector Database
            if os.path.exists(PERSIST_DIRECTORY):
                self.vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, 
                                       embedding_function=self.embedding_model)
            else:
                self.vectordb = None 

            # 3. Initialize LLM
            self.llm = ChatOllama(
                model="qwen2.5-coder:3b", 
                temperature=0.3,
                keep_alive="5m"
            )
        except Exception as e:
            st.error(f"Failed to connect to Ollama. Ensure Ollama is running and models are pulled. Error: {e}")
            self.vectordb = None
            self.llm = None

    def ingest_documents(self, file_path):
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            if self.vectordb is None:
                self.vectordb = Chroma.from_documents(
                    documents=chunks, 
                    embedding=self.embedding_model, 
                    persist_directory=PERSIST_DIRECTORY
                )
            else:
                self.vectordb.add_documents(chunks)
                
            return f"Successfully processed {len(chunks)} chunks using Ollama embeddings."
        except Exception as e:
            return f"Error during ingestion: {e}"

    def query_system(self, query):
        if not self.vectordb or not self.llm:
            return "System is not properly initialized. Check Ollama status.", []

        try:
            retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            
            response = qa_chain.invoke({"query": query})
            return response['result'], response['source_documents']
        except Exception as e:
            return f"Error during query: {e}", []

    def self_learn(self, query, answer):
        try:
            new_knowledge = f"Question: {query}\nVerified Answer: {answer}"
            doc = Document(page_content=new_knowledge, metadata={"source": "user_feedback"})
            self.vectordb.add_documents([doc])
            return "System updated! I will remember this answer for next time."
        except Exception as e:
            return f"Error during self-learning: {e}"

# ---------------------------------------------------------
# UI IMPLEMENTATION
# ---------------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 10px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        }
        .source-box {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #4facfe;
            margin-bottom: 10px;
        }
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            background: -webkit-linear-gradient(#eee, #333);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Pro-RAG | Qwen3", layout="wide")
    apply_custom_css()
    
    st.title("Pro-RAG System")
    st.markdown("### Next-Gen Retrieval Augmented Generation with Qwen3")

    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSystem()

    with st.sidebar:
        st.header("Knowledge Hub")
        st.info("Upload documents to build your project's brain.")
        uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])
        
        if uploaded_file:
            with open("temp_data.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Ingest Wisdom"):
                with st.spinner("Processing knowledge chunks..."):
                    msg = st.session_state.rag.ingest_documents("temp_data.txt")
                    if "Successfully" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Intelligence Portal")
        user_query = st.text_input("Pose your query to the system:", placeholder="What details can I find in the document?")
        
        if st.button("Generate Insight"):
            if user_query:
                with st.spinner("Synthesizing context and reasoning..."):
                    answer, sources = st.session_state.rag.query_system(user_query)
                    st.markdown("#### Generated Answer:")
                    st.write(answer)
                    st.session_state['last_query'] = user_query
                    st.session_state['last_answer'] = answer
                    st.session_state['last_sources'] = sources

    with col2:
        st.subheader("Contextual Anchors")
        if 'last_sources' in st.session_state and st.session_state['last_sources']:
            for i, doc in enumerate(st.session_state['last_sources']):
                with st.container():
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Relational Chunk {i+1}</strong><br>
                        <small>{doc.page_content[:400]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Retrieve an answer to see the source context here.")
        
        st.markdown("---")
        if 'last_answer' in st.session_state:
            st.markdown("#### Feedback & Learning")
            st.write("Is this insight accurate?")
            if st.button("Confirm & Learn"):
                with st.spinner("Updating neural weights (DB injection)..."):
                    msg = st.session_state.rag.self_learn(
                        st.session_state['last_query'], 
                        st.session_state['last_answer']
                    )
                    st.success(msg)


if __name__ == "__main__":
    main()
