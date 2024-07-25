import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Try different encodings until you find the correct one
try:
    datas = pd.read_csv("dataset.csv", encoding='utf-8')
except UnicodeDecodeError:
    datas = pd.read_csv("dataset.csv", encoding='latin-1')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(datas['CHAT'])

clf = DecisionTreeClassifier()
clf.fit(X, datas['REPLY'])

st.title("Chatbot")

user_input = st.text_input("Ask me: ", "")

if user_input:
    user_input_vector = vectorizer.transform([user_input])
    response = clf.predict(user_input_vector)
    st.write(f"Bot: {response[0]}")
