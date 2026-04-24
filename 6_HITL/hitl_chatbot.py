from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import  add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
import random
import time

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Dummy: Fetch latest stock price for a given symbol.
    Simulates API response.
    """
    time.sleep(0.5)  # simulate latency

    price = round(random.uniform(100, 500), 2)

    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": "USD",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "mock"
    }


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Dummy: Simulate stock purchase with human approval step.
    """

    # Simulated human-in-the-loop (replace with interrupt if needed)
    decision = input(f"Approve buying {quantity} shares of {symbol}? (yes/no): ")

    if decision.lower() != "yes":
        return {
            "status": "cancelled",
            "symbol": symbol,
            "quantity": quantity,
            "message": "User rejected the purchase"
        }

    price = round(random.uniform(100, 500), 2)
    total = round(price * quantity, 2)

    return {
        "status": "completed",
        "symbol": symbol.upper(),
        "quantity": quantity,
        "price_per_share": price,
        "total_cost": total,
        "currency": "USD",
        "message": "Purchase successful (mock)"
    }

tool_list = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tool_list)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tool_list)

# checkpointer
memory = MemorySaver()

# graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges(
    "chat", tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer = memory)

chatbot



if __name__ == "__main__":

    # Use a fixed thread_id so the conversation is persisted in memory

    thread_id = "demo_thread"

    while True:
        
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content = user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(
            state,
            config = {'configurable': {"thread_id": thread_id}},
        )

        # Check for HITL interrupt from purchase_stock
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume = decision),
                config = {"configurable": {"thread_id": thread_id}}
            )

        # Get the latest message from the assistant
        messages = result['messages']
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")