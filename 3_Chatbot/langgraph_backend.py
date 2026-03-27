from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3

load_dotenv()

model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

class ChatState(TypedDict):
    chat: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:

    messages = state['chat']

    response = model.invoke(messages)

    return {'chat' : [response]}

conn = sqlite3.connect(database = 'chatbot.db', check_same_thread = False)
# Checkpointer
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)

# to check threads in checkpointer
def retrieve_all_threads():
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])
        return (list(all_threads))

def delete_thread_from_db(thread_id):
    cursor = conn.cursor()
    tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()

if __name__ == "__main__":
    CONFIG = {'configurable': {'thread_id': 'thread-2'}}

    response = chatbot.invoke(
        {'chat': [HumanMessage(content = 'What is the recipe of pizza?')]},
        config = CONFIG,
    )

    print(response['chat'][-1].content)

    # # for message_chunk, metadata in chatbot.stream(
    # #     {'chat': [HumanMessage(content = 'What is the recipe of pasta?')]},
    # #     config = CONFIG,
    # #     stream_mode = 'messages' 
    # #     ):

    # #     if message_chunk.content:
    # #         print(message_chunk.content, end = " ", flush = True)