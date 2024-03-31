import streamlit as st
from model import *

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>Chunking</h1> <br><h3>Enter sentence below:</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    model = Model()
    epochs = 3

    model.load_weights()

    user_input = st.text_input("")
    caption_button = st.button("Predict Chunks")
    if caption_button:
        st.balloons()
        if user_input:
            pos_tags, chunks, sentence = model.inference(user_input)
            print(sentence, chunks)
            st.markdown(f"<h2 style='text-align: center;'>{chunks}</h2>",
                            unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{sentence}</h2>",
                            unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{pos_tags}</h2>",
                            unsafe_allow_html=True)
        else:
            print("Enter a sentence to do prediction!")