from langchain_core.messages import HumanMessage
import streamlit as st
from langgraph_backend import chatbot

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# st. session_state -> dict -›
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# loading the conversation history
for chat in st.session_state.chat_history:
    with st.chat_message(chat['role']):
        st.text(chat['content'])

user_input = st.chat_input('Type here...')

if user_input:
    # add user message to history
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)
    
    response = chatbot.invoke({'conversation': [HumanMessage(content = user_input)]}, config = CONFIG)
    ai_message = response['conversation'][-1].content
    
    # add assistant message to history
    st.session_state.chat_history.append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)# with st.chat_message('user'):