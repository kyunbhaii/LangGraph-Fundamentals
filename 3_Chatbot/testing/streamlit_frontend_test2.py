from langchain_core.messages import HumanMessage
import streamlit as st

st.set_page_config(layout="wide")

from langgraph_backend import chatbot
import uuid

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['chat_history'] = []
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        if 'thread_titles' not in st.session_state:
            st.session_state['thread_titles'] = {}
        if thread_id not in st.session_state['thread_titles']:
             st.session_state['thread_titles'][thread_id] = "New Chat"

def load_conversation(thread_id):
    state = chatbot.get_state(config = {'configurable': {'thread_id': thread_id}})
    return state.values.get('chat', []) if state.values else []

# st. session_state -> dict -›
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

add_thread(st.session_state['thread_id'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

user_input = st.chat_input('Type here...')

if user_input:
    # Update title instantly before sidebar rendering
    current_thread_id = st.session_state['thread_id']
    if st.session_state['thread_titles'].get(current_thread_id) == "New Chat":
        st.session_state['thread_titles'][current_thread_id] = user_input[:30] + ("..." if len(user_input) > 30 else "")

# Sidebar UI
st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][:: -1]:
    title = st.session_state['thread_titles'].get(thread_id, "New Chat")
    
    # Only display threads that have an actual recorded name (not empty "New Chat"s)
    if title != "New Chat":
        if st.sidebar.button(title, key=thread_id):
            st.session_state['thread_id'] = thread_id
            chats = load_conversation(thread_id)

            temp_chats = []

            for chat_msg in chats:
                if isinstance(chat_msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_chats.append({'role': role, 'content': chat_msg.content})
            st.session_state['chat_history'] = temp_chats
            st.rerun()

# loading the conversation history
if len(st.session_state.chat_history) == 0 and not user_input:
    st.title("New Chat")
    st.markdown("Welcome! Type a message below to start a new conversation.")
elif len(st.session_state.chat_history) > 0:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])

if user_input:
    # add user message to history
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)
    
    # add assistant message to history
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'chat': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = 'messages'
            ) if message_chunk.content
        )
    st.session_state.chat_history.append({'role': 'assistant', 'content': ai_message})