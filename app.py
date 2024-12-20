# app.py
import streamlit as st
import os
import sys

try:
    from chatbot import run_chatbot
except Exception as e:
    st.error(f"Error loading chatbot: {str(e)}")
    st.stop()

st.title("Scrum Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about Scrum..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Get chatbot response
        response = run_chatbot(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
