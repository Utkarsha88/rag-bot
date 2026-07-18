import os
import streamlit as st

# 1. Page Config MUST be the absolute first Streamlit command
st.set_page_config(page_title="Enterprise RAG Bot", page_icon="🤖", layout="wide")
st.title("🤖 Document Intelligence RAG Assistant")

# 2. Try loading dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ python-dotenv is not installed. Run: pip install python-dotenv")

# 3. Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")
    st.stop()

# 4. Safe Heavy Imports
with st.spinner("Loading AI models..."):
    try:
        from langchain_community.document_loaders import PDFPlumberLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as e:
        st.error(f"❌ Missing a required library: {e}")
        st.info("Make sure you installed everything in this environment: pip install langchain langchain-google-genai langchain-huggingface langchain-community langchain-classic chromadb pdfplumber")
        st.stop()

# --- Everything below this line is the same as before ---
st.markdown("Upload documents in the sidebar and ask questions below.")

UPLOAD_DIR = "uploaded_docs"
DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF dataset", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved: {uploaded_file.name}")
        
        if st.button("🚀 Index Document", use_container_width=True):
            with st.spinner("Processing document structures..."):
                try:
                    loader = PDFPlumberLoader(file_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=150,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(docs)
                    
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=DB_DIR
                    )
                    
                    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)
                    retriever = vector_db.as_retriever(search_kwargs={"k": 12})
                    
                    system_prompt = (
                        "You are a professional assistant analyzing documents.\n"
                        "Use the following pieces of retrieved context to answer the user's question.\n"
                        "If you cannot find the answer in the context, clearly state that you don't know.\n"
                        "Do not make up information.\n\n"
                        "Context:\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    qa_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)
                    
                    st.balloons()
                    st.success("Database fully updated and ready!")
                except Exception as e:
                    st.error(f"Error during ingestion pipeline: {str(e)}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your dataset..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        if st.session_state.rag_chain is None:
            response_text = "⚠️ Please upload and index a document in the sidebar before asking questions."
            st.markdown(response_text)
        else:
            with st.spinner("Analyzing documents..."):
                try:
                    res = st.session_state.rag_chain.invoke({"input": user_query})
                    response_text = res["answer"]
                    st.markdown(response_text)
                except Exception as e:
                    response_text = f"❌ Error generating answer: {str(e)}"
                    st.error(response_text)
                    
    st.session_state.messages.append({"role": "assistant", "content": response_text})