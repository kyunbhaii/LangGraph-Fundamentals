from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

class ChatState(TypedDict):
    chat: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:

    messages = state['chat']

    response = model.invoke(messages)

    return {'chat' : [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)

if __name__ == "__main__":
    CONFIG = {'configurable': {'thread_id': 'thread-1'}}

    response = chatbot.invoke(
        {'chat': [HumanMessage(content = 'What is the recipe of pasta?')]},
        config = CONFIG,
    )

    print(chatbot.get_state(config=CONFIG).values['chat'])

    # for message_chunk, metadata in chatbot.stream(
    #     {'chat': [HumanMessage(content = 'What is the recipe of pasta?')]},
    #     config = CONFIG,
    #     stream_mode = 'messages' 
    #     ):

    #     if message_chunk.content:
    #         print(message_chunk.content, end = " ", flush = True)