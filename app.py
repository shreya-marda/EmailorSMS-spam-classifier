import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', mode='rb'))  # readbinary mode
model = pickle.load(open('model.pkl', mode='rb'))

st.title("Email/SMS spam classifier")

input_sms = st.text_area("Enter the message")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    lst = []
    for i in text:
        if i.isalnum():
            lst.append(i)

    text = lst[:]
    lst.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            lst.append(i)

    text = lst[:]
    lst.clear()

    for i in text:
        i = ps.stem(i)
        lst.append(i)
    return " ".join(lst)

if st.button("Predict"):
    # 1. preprocess
    transformed_msg = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_msg]) 
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if (result == 1):
        st.header("Spam")
    else:
        st.header("Not spam")