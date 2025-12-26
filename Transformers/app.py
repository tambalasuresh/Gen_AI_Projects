import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Transformer AI App", layout="centered")

st.title("🤖 Transformer AI with Streamlit")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    models = {
        "sentiment": pipeline("sentiment-analysis"),
        "text_gen": pipeline("text-generation", model="gpt2"),
        "qa": pipeline("question-answering"),
        "speech": pipeline("automatic-speech-recognition", model="openai/whisper-small")
    }
    return models

models = load_models()

# ================= SELECT TASK =================
task = st.selectbox(
    "Select Transformer Task",
    ["Sentiment Analysis", "Text Generation", "Question Answering", "Speech Recognition"]
)

# ================= SENTIMENT ANALYSIS =================
if task == "Sentiment Analysis":
    text = st.text_area("Enter text")

    if st.button("Analyze"):
        result = models["sentiment"](text)[0]
        st.success(f"Sentiment: {result['label']}")
        st.write(f"Confidence: {result['score']:.4f}")

# ================= TEXT GENERATION =================
elif task == "Text Generation":
    prompt = st.text_area("Enter prompt")

    if st.button("Generate Text"):
        output = models["text_gen"](prompt, max_length=100, num_return_sequences=1)
        st.write(output[0]["generated_text"])

# ================= QUESTION ANSWERING =================
elif task == "Question Answering":
    context = st.text_area("Context", placeholder="Paste paragraph here")
    question = st.text_input("Question")

    if st.button("Get Answer"):
        result = models["qa"](question=question, context=context)
        st.success(f"Answer: {result['answer']}")
        st.write(f"Score: {result['score']:.4f}")

# ================= SPEECH RECOGNITION =================
elif task == "Speech Recognition":
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        with st.spinner("Transcribing..."):
            text = models["speech"](audio_file)
            st.success("Transcription:")
            st.write(text["text"])
