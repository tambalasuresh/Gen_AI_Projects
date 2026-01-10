import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone

# ========== CONFIGURATION ==========
PINECONE_API_KEY = "pcsk_6srVnS_6GCXMTgsAPkTytddZnPmBQUob5MGHussED57QEH6LkbWhAqw6iWnCqogn9akNQ4"
INDEX_NAME = "rag-demo"
ENVIRONMENT = "https://rag-demo-v8zxvpj.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
GENERATION_MODEL_NAME = "google/flan-t5-base"
TOP_K = 3
# ===================================

# --- Caching Model Loads ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model=GENERATION_MODEL_NAME)

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=ENVIRONMENT)
    return pc.Index(INDEX_NAME)

# --- RAG Pipeline ---
def generate_answer(query, embedder, index, generator):
    query_vector = embedder.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    if not results["matches"]:
        return "No relevant context found.", ""

    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    prompt = f"""Context:
{context}

Question: {query}
Answer:"""

    output = generator(prompt, max_new_tokens=256)[0]['generated_text']
    return output, context

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="RAG with Pinecone & Flan-T5", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (RAG)")
    st.markdown("Ask a question and get answers using Flan-T5 with Pinecone vector search.")

    query = st.text_input("🔎 Enter your question here:")

    if query:
        with st.spinner("Searching and generating answer..."):
            embedder = load_embedder()
            generator = load_generator()
            index = init_pinecone()

            answer, context = generate_answer(query, embedder, index, generator)

        st.subheader("✅ Generated Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            st.code(context)

# Run the app
if __name__ == "__main__":
    main()
