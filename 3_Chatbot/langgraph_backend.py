from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
import asyncio
import httpx
import aiosqlite

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler
from langfuse import observe, propagate_attributes
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
 
load_dotenv()

# Standard Langfuse initialization
langfuse_handler = CallbackHandler()

model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# MCP Client Configuration
client = MultiServerMCPClient(
    {
        "ChatbotTools": {
            "transport": "stdio",
            "command": "/Users/vikrmaditya/Downloads/Code/LangGraph/venv/bin/python3",
            "args": ["/Users/vikrmaditya/Downloads/Code/LangGraph/3_Chatbot/mcp_server.py"]
        }
    }
)

async def build_graph(conn):

    # Retrieve tools from MCP server dynamically
    all_tools = await client.get_tools()
    llm_with_tools = model.bind_tools(tools = all_tools)
    print(all_tools)

    # graph_nodes defined inside to closure over llm_with_tools
    async def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:
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

        # We pass the metadata via the config, which Langfuse's CallbackHandler picks up
        response = await llm_with_tools.ainvoke(
            [system] + messages,
            config=config 
        )

        return {'messages': [response]}

    tool_node = ToolNode(all_tools)

    # Checkpointer
    checkpointer = AsyncSqliteSaver(conn = conn)

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

    return chatbot

# to check threads in checkpointer
async def retrieve_all_threads(checkpointer):
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return (list(all_threads))

async def delete_thread_from_db(thread_id):
    async with aiosqlite.connect('chatbot.db') as conn:
        tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
        for table in tables:
            try:
                await conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except aiosqlite.OperationalError:
                pass
        await conn.commit()

async def main():
    CONFIG = {
        'run_name': 'Chatbot_Local_Test',
        'tags': ['Project: LangGraph Chatbot', 'Local-Test'],
        'metadata': {'project': 'LangGraph Chatbot', 'model': 'llama-3.1-8b-instant', 'model_temp': '0.3'},
        'configurable': {'thread_id': 'thread-2'},
        'recursion_limit': 20,
        'callbacks': [langfuse_handler]
    }

    async with aiosqlite.connect("chatbot.db") as conn:
        chatbot = await build_graph(conn)

        response = await chatbot.ainvoke(
            {'messages': [HumanMessage(content = 'What are the key benefits of the HDFC Optima Policy?')]},
            config = CONFIG,
        )

        print(response['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main())