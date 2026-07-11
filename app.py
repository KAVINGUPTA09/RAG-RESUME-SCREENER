import os
import uuid
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def generate_embeddings(self, text):
        return self.model.encode(text, show_progress_bar=False)

class VectorStoreManager:
    def __init__(self, persist_directory="data/resume-store", collection_name="resume_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._initiate_store()

    def _initiate_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Mismatch")
        ids, all_metadata, document_content, embeddings_list = [], [], [], []
        for i, doc in enumerate(documents):
            ids.append(f"doc_{uuid.uuid4()}")
            metadata = dict(doc.metadata)
            all_metadata.append(metadata)
            document_content.append(doc.page_content)
            embeddings_list.append(embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else embeddings[i])

        batch_size = 4000
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(ids=ids[i:end_idx], metadatas=all_metadata[i:end_idx], documents=document_content[i:end_idx], embeddings=embeddings_list[i:end_idx])

class RAG_Retriver:
    def __init__(self, store_manager, embedding_manager):
        self.store_manager = store_manager
        self.embedding_manager = embedding_manager
    def retrive(self, query, top_k=3):
        query_vector = self.embedding_manager.generate_embeddings([query])[0]
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        results = self.store_manager.collection.query(query_embeddings=[query_vector], n_results=top_k)
        retrived_docs = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, metadata, document, distance) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0])):
                retrived_docs.append({"metadata": metadata, "text": document})
        return retrived_docs

st.set_page_config(page_title="TalentAlign AI Dashboard", layout="wide")

st.title("🎯 TalentAlign AI: Enterprise Resume Screener")
st.write("Advanced Semantic Matching System using **RAG Architecture** (ChromaDB + Llama 3.1).")

m1, m2, m3 = st.columns(3)
m1.metric("Target Organization", "Enterprise")
m2.metric("Database Engine", "ChromaDB")
m3.metric("LLM Orchestration", "Llama 3.1")

st.write("---")

if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()
if 'store_manager' not in st.session_state:
    st.session_state.store_manager = VectorStoreManager()

with st.sidebar:
    st.header("📂 Ingestion Workspace")
    docx_folder = "./resumes"
    os.makedirs(docx_folder, exist_ok=True)
    uploaded_files = st.file_uploader("Upload Word Resumes (.docx)", type=["docx"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(docx_folder, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Loaded {len(uploaded_files)} resumes!")

    if st.button("⚡ Vectorize Assets"):
        documents = []
        if os.path.exists(docx_folder):
            for file_name in os.listdir(docx_folder):
                if file_name.endswith(".docx"):
                    loader = Docx2txtLoader(os.path.join(docx_folder, file_name))
                    single_docs = loader.load()
                    for doc in single_docs:
                        doc.metadata["candidate_name"] = file_name.replace(".docx", "")
                    documents.extend(single_docs)
        
        if documents:
            with st.spinner("Embedding segments..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
                chunked_docs = text_splitter.split_documents(documents)
                texts = [doc.page_content for doc in chunked_docs]
                embeddings = st.session_state.embedding_manager.generate_embeddings(texts)
                st.session_state.store_manager.add_documents(chunked_docs, embeddings)
                st.success("Vector Matrix Built!")

job_description = st.text_area("📋 Requirements Matrix / Job Description:", height=150)
top_k_select = st.slider("Max Candidates Filters:", 1, 5, 3)

if st.button("🔍 Execute Semantic Alignment Scan"):
    if not job_description.strip():
        st.warning("Please specify criteria.")
    else:
        with st.spinner("Parsing neural nodes..."):
            rag_retriever = RAG_Retriver(st.session_state.store_manager, st.session_state.embedding_manager)
            results = rag_retriever.retrive(job_description, top_k=top_k_select)
            
            context = "\n\n".join([f"Candidate: {doc['metadata'].get('candidate_name', 'Unknown')}\nText:\n{doc['text']}" for doc in results])
            
            if context:
                # 🔥 FIX: Yahan secrets injection system kar diya hai production standard ke liye
                groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
                if not groq_key:
                    st.error("GROQ_API_KEY missing in Streamlit Secrets setup.")
                    st.stop()
                llm = ChatGroq(groq_api_key=groq_key, temperature=0.3, model="llama-3.1-8b-instant")
                prompt = f"Context:{context}\nQuery:{job_description}\nReturn ONLY a structured Markdown Table with Rank, Candidate Name, Match Score (1-100), Fitment Reason."
                response = llm.invoke(prompt)
                
                st.subheader("📊 Candidate Evaluation Dashboard Matrix")
                st.markdown(response.content)
            else:
                st.error("Database empty.")
