from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler
from langfuse import observe, propagate_attributes
import sqlite3
import os
import requests
 
load_dotenv()

# Standard Langfuse initialization -- it picks up keys from .env automatically
langfuse_handler = CallbackHandler()

model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Tools
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool(description="Perform arithmetic operations: add, sub, mul, div on two numbers")
@observe(as_type='tool')
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    try:
        if operation == 'add':
            result = first_num + second_num
        elif operation == 'sub':
            result = first_num - second_num
        elif operation == 'mul':
            result = first_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}

@tool(description="Fetch latest stock price for a given symbol")
@observe(as_type='tool')
def get_stock_price(symbol: str) -> dict:
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=VJ5UT2AO4KBQKZJT"

    try:
        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            return {"error": f"API request failed with status {r.status_code}"}

        data = r.json()

        # Handle rate limit
        if "Note" in data:
            return {"error": "API rate limit reached"}

        price = data.get("Global Quote", {}).get("05. price")

        if not price:
            return {"error": "Stock price not available"}

        return {
            "symbol": symbol.upper(),
            "price": float(price)
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}

    except Exception as e:
        return {"error": str(e)}
    
my_tools = [calculator, get_stock_price, search_tool]
llm_with_tools = model.bind_tools(tools = my_tools)

# graph_nodes
@observe()
def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:

    with propagate_attributes(
        tags=["Project: LangGraph Chatbot"],
        metadata={"tools": "calculator, stock_price, search"}
    ):
        messages = state['messages']

        system = SystemMessage(content = """
            You are a tool-using assistant.

            Rules:
            - Use tools only when necessary
            - Do NOT call the same tool repeatedly
            - If you already have enough information, return final answer
            - Do NOT hallucinate additional data
        """
        )

        response = llm_with_tools.invoke(
            [system] + messages,
            config=config # Pass the full config which already contains callbacks
        )

    return {'messages': [response]}

tool_node = ToolNode(my_tools)

conn = sqlite3.connect(database = 'chatbot.db', check_same_thread = False)
# Checkpointer
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
        'chat_node', 
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
graph.add_edge('tools', 'chat_node')

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
    CONFIG = {
        'run_name': 'Chatbot_Local_Test',
        'tags': ['Project: LangGraph Chatbot', 'Local-Test'],
        'metadata': {'project': 'LangGraph Chatbot', 'model': 'llama-3.1-8b-instant', 'model_temp': '0.3'},
        'configurable': {'thread_id': 'thread-2'},
        'recursion_limit': 5,
        'callbacks': [langfuse_handler]
    }

    response = chatbot.invoke(
        {'messages': [HumanMessage(content = 'What is the recipe of pizza?')]},
        config = CONFIG,
    )

    print(response['messages'][-1].content)


    # # for message_chunk, metadata in chatbot.stream(
    # #     {'chat': [HumanMessage(content = 'What is the recipe of pasta?')]},
    # #     config = CONFIG,
    # #     stream_mode = 'messages' 
    # #     ):

    # #     if message_chunk.content:
    # #         print(message_chunk.content, end = " ", flush = True)