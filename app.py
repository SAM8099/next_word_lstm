import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('saved_model/model.h5')

with open('tokenizer.pkl', 'rb') as file:
    token = pickle.load(file)

def predict_next(model, token, text, max_seq):
    token_list = token.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq:
        token_list = token_list[-(max_seq-1):]
    token_list = pad_sequences([token_list], maxlen=max_seq-1, padding='pre')
    prediction = model.predict(token_list, verbose=0)
    predict_next_word = np.argmax(prediction, axis=1)
    for word, index in token.word_index.items():
        if index == predict_next_word:
            return word
    return None

st.title("Next word prediction using lstm")
input_text = st.text_input("write text")
if st.button("Predict"):
    max_seq_len = model.input_shape[1]+1
    next_word = predict_next(model, token, input_text, max_seq_len)
    st.write("Next word is: ", next_word)

