from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import asyncio
import aiosqlite

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* ---------------- SIDEBAR LOCKED ---------------- */

/* Fixed width */
[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
    width: 320px !important;
}

/* Prevent collapse animation */
section[data-testid="stSidebar"] {
    transform: none !important;
}

/* ---------------- REMOVE ALL TOGGLE BUTTONS ---------------- */

/* Old toggle button */
button[kind="sidebarToggle"] {
    display: none !important;
}

/* New toggle button (top arrow icon) */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Collapsed expand button */
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Sometimes appears as generic button in header */
header button {
    display: none !important;
}

/* Extra safety: prevent hidden state */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    visibility: visible !important;
}

/* Keep sidebar content stable */
[data-testid="stSidebarContent"] {
    min-width: 320px !important;
}

</style>
""", unsafe_allow_html=True)

from langgraph_backend import build_graph, retrieve_all_threads, delete_thread_from_db, langfuse_handler
from langfuse import observe, propagate_attributes
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
        # Insert at the beginning (index 0) so new chats are at the top
        st.session_state['chat_threads'].insert(0, thread_id)
        if 'thread_titles' not in st.session_state:
            st.session_state['thread_titles'] = {}
        if thread_id not in st.session_state['thread_titles']:
             st.session_state['thread_titles'][thread_id] = "New Chat"

async def load_conversation_async(thread_id):
    async with aiosqlite.connect("chatbot.db") as conn:
        chatbot = await build_graph(conn)
        state = await chatbot.aget_state(config = {'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', []) if state.values else []

def load_conversation(thread_id):
    return asyncio.run(load_conversation_async(thread_id))

# st. session_state -> dict -›
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    async def get_initial_threads():
        async with aiosqlite.connect("chatbot.db") as conn:
            checkpointer = AsyncSqliteSaver(conn=conn)
            return await retrieve_all_threads(checkpointer)
    st.session_state['chat_threads'] = asyncio.run(get_initial_threads())

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

def get_config():
    return {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'run_name': 'Chatbot_Interaction',
        'tags': ['Project: LangGraph Chatbot', 'Streamlit', st.session_state['thread_id']],
        'metadata': {
            'project': 'Chatbot',
            'session_id': st.session_state['thread_id'],
            'model': 'llama-3.1-8b-instant', 
            'model_temp': '0.3', 
            'thread_id': st.session_state['thread_id']
        },
        'recursion_limit': 5,
        'callbacks': [langfuse_handler]
    }



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

for thread_id in st.session_state['chat_threads']:
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
                    # Skip ToolMessages and AIMessages that are just tool calls (no text content)
                    if isinstance(chat_msg, ToolMessage):
                        continue
                    if isinstance(chat_msg, AIMessage) and chat_msg.tool_calls and not chat_msg.content:
                        continue
                        
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
                if st.button("Rename", key=f"rename_{thread_id}", use_container_width=True):
                    if st.session_state.get('editing_thread') == thread_id:
                        st.session_state['editing_thread'] = None
                    else:
                        st.session_state['editing_thread'] = thread_id
                    st.rerun()
                    
                if st.button("Delete", key=f"delete_{thread_id}", use_container_width=True):
                    asyncio.run(delete_thread_from_db(thread_id))
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

@observe(name="Chatbot")
def generate_response(user_input):
    with propagate_attributes(
        session_id=st.session_state['thread_id'],
        tags=["Project: LangGraph Chatbot", st.session_state['thread_id']],
        metadata={"project": "LangGraph Chatbot", "thread_id": st.session_state['thread_id']}
    ):

        # add assistant message to history
        with st.chat_message('assistant'):
            
            async def stream_generator():
                async with aiosqlite.connect("chatbot.db") as conn:
                    chatbot = await build_graph(conn)
                    status_container = None
                    
                    async for message_chunk, metadata in chatbot.astream(
                        {'messages': [HumanMessage(content=user_input)]},
                        config = get_config(),
                        stream_mode = 'messages'
                    ):
                        # 1. Detected a tool call intent
                        if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                            status_container = st.status("Thinking...", expanded=False)
                            for tool_call in message_chunk.tool_calls:
                                status_container.write(f"Using tool: `{tool_call['name']}`")
                                status_container.update(label=f"Running `{tool_call['name']}`...")
                        
                        # 2. Detected a tool response
                        if isinstance(message_chunk, ToolMessage):
                            if status_container:
                                status_container.write(f"Tool `{message_chunk.name}` finished.")
                                status_container.update(label="✅ Tool finished", state="complete", expanded=False)

                        # 3. Stream the actual text response
                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            yield message_chunk.content

            ai_message = st.write_stream(stream_generator())

        st.session_state.chat_history.append({'role': 'assistant', 'content': ai_message})
        return ai_message

if user_input:
    # add user message to history
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Process and stream the response
    generate_response(user_input)