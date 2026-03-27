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
    
    # add assistant message to history
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'conversation': [HumanMessage(content=user_input)]},
                config = {'configurable': {'thread_id': 'thread-1'}},
                stream_mode = 'messages'
            ) if message_chunk.content
        )
    
    st.session_state.chat_history.append({'role': 'assistant', 'content': ai_message})