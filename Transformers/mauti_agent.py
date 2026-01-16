import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Multi-Agent NLP App", layout="centered")
st.title("🤖 Multi-Agent Streamlit App")

# Load models
@st.cache_resource
def load_agents():
    agents = {
        "sentiment": pipeline("sentiment-analysis"),
        "summarizer": pipeline("summarization"),
        "generator": pipeline("text-generation", model="gpt2")
    }
    return agents

agents = load_agents()

# User input
user_text = st.text_area("Enter your text", height=100)

# Button is disabled if no text is entered
run_button = st.button(
    "Run All Agents",
    disabled=not bool(len(user_text) > 0)
)

if run_button:

    st.subheader("🧠 Agent Outputs")

    # Sentiment Agent
    sentiment_result = agents["sentiment"](user_text)[0]
    st.markdown("### 😊 Sentiment Agent")
    st.write(sentiment_result)

    # Summarizer Agent
    if len(user_text.split()) > 30:
        summary = agents["summarizer"](
            user_text, max_length=60, min_length=25
        )[0]["summary_text"]
        st.markdown("### ✂️ Summarizer Agent")
        st.write(summary)

    # Generator Agent
    generated = agents["generator"](user_text, max_length=60)[0]["generated_text"]
    st.markdown("### ✍️ Generator Agent")
    st.write(generated)
