import streamlit as st
from transformers import pipeline 


st.set_page_config(page_title="Suresh Practice",layout="centered")
st.title("This is Trnasformer AI Project")

@st.cache_resource
def load_model():
    model ={
        "sentiment":pipeline("sentiment-analysis"),
        "text_gen": pipeline("text-generation", model="gpt2")
    }
    return model 

model = load_model()

task = st.selectbox("Select Your Model",["Sentiment-Analysis","Text Geanration"])

# st.write("Enter Your Query")


if task ==  "Sentiment-Analysis":
    user_query = st.text_area("Enter Your Text")

    if st.button("Analyze") and user_query:
        sentiment = model["sentiment"](user_query)[0]
        if sentiment["label"] == "POSITIVE":
            st.success(sentiment)
        else:
            st.warning(sentiment)
else:
    promt = st.text_input("Enter Yout Promt")

    if st.button("Geanrate Text") and promt:
        result = model["text_gen"](promt,max_length = 250,num_return_sequences = 1)
        # st.write(result[0])
        st.write(result[0]["generated_text"])

