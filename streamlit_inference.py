import streamlit as st
from model import *

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>Chunking</h1> <br><h3>Enter URL Link Below:</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    model = Model()
    epochs = 3

    model.load_weights()

    user_input = st.text_input("")
    caption_button = st.button("Predict")
    if caption_button:
        st.balloons()
        if user_input:
            chunks, sentence = model.inference(user_input)
            print(sentence, chunks)
            st.markdown(f"<h2 style='text-align: center;'>{sentence}</h2>",
                            unsafe_allow_html=True)
        else:
            print("Enter a sentence for doing prediction")