from langchain_core.messages import HumanMessage
import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
    /* Main content width */
    .block-container {
        max-width: 900px !important;
        margin: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        max-width: 100% !important;
    }

    /* Chat input container fix */
    [data-testid="stChatInput"] {
        max-width: 900px !important;
        margin: auto !important;
    }

    /* Input box inside */
    [data-testid="stChatInput"] > div {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

from langgraph_backend import chatbot, retrieve_all_threads, delete_thread_from_db
import uuid
import json
import os

TITLES_FILE = "chat_titles.json"

def load_titles():
    if os.path.exists(TITLES_FILE):
        with open(TITLES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_titles(titles):
    with open(TITLES_FILE, 'w') as f:
        json.dump(titles, f)

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
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = load_titles()
    for t_id in st.session_state['chat_threads']:
        if t_id not in st.session_state['thread_titles']:
            chats = load_conversation(t_id)
            if chats:
                for msg in chats:
                    if isinstance(msg, HumanMessage):
                        title = msg.content[:30] + ("..." if len(msg.content) > 30 else "")
                        st.session_state['thread_titles'][t_id] = title
                        save_titles(st.session_state['thread_titles'])
                        break

if 'editing_thread' not in st.session_state:
    st.session_state['editing_thread'] = None

add_thread(st.session_state['thread_id'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

user_input = st.chat_input('Type here...')

if user_input:
    # Update title instantly before sidebar rendering
    current_thread_id = st.session_state['thread_id']
    if st.session_state['thread_titles'].get(current_thread_id) == "New Chat":
        st.session_state['thread_titles'][current_thread_id] = user_input[:30] + ("..." if len(user_input) > 30 else "")
        save_titles(st.session_state['thread_titles'])

# Sidebar UI
st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][:: -1]:
    title = st.session_state['thread_titles'].get(thread_id, "New Chat")
    
    # Only display threads that have an actual recorded name (not empty "New Chat"s)
    if title != "New Chat":
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            if st.button(title, key=f"btn_{thread_id}", use_container_width=True):
                st.session_state['thread_id'] = thread_id
                chats = load_conversation(thread_id)
                temp_chats = []
                for chat_msg in chats:
                    role = 'user' if isinstance(chat_msg, HumanMessage) else 'assistant'
                    temp_chats.append({'role': role, 'content': chat_msg.content})
                st.session_state['chat_history'] = temp_chats
                st.rerun()
        
        with col2:
            if st.button("⌄", key=f"menu_{thread_id}"):
                if st.session_state.get('active_menu') == thread_id:
                    st.session_state['active_menu'] = None
                else:
                    st.session_state['active_menu'] = thread_id
                st.session_state['editing_thread'] = None
                st.rerun()
                
        if st.session_state.get('active_menu') == thread_id:
            with st.sidebar.container(border=True):
                if st.button("Rename ✏️", key=f"rename_{thread_id}", use_container_width=True):
                    if st.session_state.get('editing_thread') == thread_id:
                        st.session_state['editing_thread'] = None
                    else:
                        st.session_state['editing_thread'] = thread_id
                    st.rerun()
                    
                if st.button("Delete 🗑️", key=f"delete_{thread_id}", use_container_width=True):
                    delete_thread_from_db(thread_id)
                    if thread_id in st.session_state['chat_threads']:
                        st.session_state['chat_threads'].remove(thread_id)
                    if thread_id in st.session_state['thread_titles']:
                        del st.session_state['thread_titles'][thread_id]
                        save_titles(st.session_state['thread_titles'])
                    if st.session_state['thread_id'] == thread_id:
                        reset_chat()
                    st.session_state['active_menu'] = None
                    st.rerun()
                
        if st.session_state.get('editing_thread') == thread_id:
            new_title = st.sidebar.text_input("New Name:", value=title, key=f"input_{thread_id}")
            if st.sidebar.button("Save", key=f"save_{thread_id}"):
                st.session_state['thread_titles'][thread_id] = new_title
                save_titles(st.session_state['thread_titles'])
                st.session_state['editing_thread'] = None
                st.session_state['active_menu'] = None
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