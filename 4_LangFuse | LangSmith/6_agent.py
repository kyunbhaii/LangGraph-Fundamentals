from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

# ----------------- Tools -----------------
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city."""
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return str(response.json())

tools = [search_tool, get_weather_data]

# ----------------- LLM + Prompt -----------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Modern LangGraph agent — uses native tool calling (no text parsing, no hub prompt)
agent = create_react_agent(
    model=llm,
    tools=tools,
)

# ----------------- Traced Runner -----------------
@observe(name="agent_run")
def run_agent(query: str) -> str:
    with propagate_attributes(
        tags=["agent", "tool_calling"],
        metadata={"model": "llama-3.1-8b-instant", "version": "v1"}
    ):
        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"callbacks": [langfuse_handler]}
        )
        return response["messages"][-1].content

# ----------------- CLI -----------------
if __name__ == "__main__":
    # What is the release date of Dhadak 2?
    # What is the current temp of gurgaon?
    # Identify the birthplace city of Kalpana Chawla and give its current temperature.

    q = "Identify the birthplace city of Kalpana Chawla and give its current temperature"
    print(f"\nQ: {q}")
    ans = run_agent(q)
    print(f"\nA: {ans}")