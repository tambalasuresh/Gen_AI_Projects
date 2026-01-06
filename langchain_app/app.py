import streamlit as st
import os

# -------- LangChain Imports (STABLE) --------
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# -------- Transformers --------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------- Streamlit UI --------
st.set_page_config(page_title="LangChain PDF/TXT Q&A", layout="wide")
st.title("📄 LangChain PDF & TXT Q&A (100% FREE)")
st.write("Upload a PDF or TXT file and ask questions")

# -------- File Upload --------
uploaded_file = st.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()

    # Save file
    with open(file_name, "wb") as f:
        f.write(uploaded_file.read())

    # -------- Load File --------
    if file_ext == "pdf":
        loader = PyPDFLoader(file_name)
    elif file_ext == "txt":
        loader = TextLoader(file_name, encoding="utf-8")

    documents = loader.load()

    # -------- Split Text --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    st.success(f"File loaded successfully ({len(docs)} chunks created)")

    # -------- Embeddings --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- Vector DB --------
    db = FAISS.from_documents(docs, embeddings)

    # -------- FREE LLM --------
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # -------- RAG Chain --------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # -------- Ask Question --------
    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking... 🤔"):
            result = qa.invoke({"query": query})

        st.subheader("✅ Answer")
        st.write(result["result"])

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)

    # Cleanup
    os.remove(file_name)
