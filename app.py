# app.py
import streamlit as st
from chatbot import run_chatbot

st.title("Scrum Chatbot")

user_input = st.text_area("You:", height=200)

if st.button("Send"):
    chatbot_response = run_chatbot(user_input)
    st.write("Chatbot:", chatbot_response)