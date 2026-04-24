from fastmcp import FastMCP
import httpx
import os
import json
import hashlib
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("ChatbotTools")
search_tool = DuckDuckGoSearchRun(region='us-en')

# RAG Configuration
PDF_PATH = "Policy_HDFC optima.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- RAG Helpers (ported from rag_v3.py) -----------------

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    if not p.exists():
        return {}
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

def build_index(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    emb = HuggingFaceEndpointEmbeddings(model=embed_model_name)
    vs = FAISS.from_documents(splits, emb)
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs

def load_or_build_index(pdf_path: str, chunk_size=1000, chunk_overlap=150, embed_model_name=EMBED_MODEL):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    
    emb = HuggingFaceEndpointEmbeddings(model=embed_model_name)
    if index_dir.exists():
        return FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
    else:
        return build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------- MCP Tools -----------------

@mcp.tool()
async def rag_tool(query: str) -> str:
    """Answers questions based on the 'Policy_HDFC optima.pdf' insurance document. 
    Use this for any insurance policy related queries.
    """
    try:
        # 1. Setup Vectorstore
        vectorstore = load_or_build_index(PDF_PATH)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # 2. Setup LLM and Prompt
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])
        
        # 3. Setup Chain
        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        chain = parallel | prompt | llm | StrOutputParser()
        
        # 4. Invoke
        return await chain.ainvoke(query)
        
    except Exception as e:
        return f"Error executing RAG tool: {str(e)}"

@mcp.tool()
async def search(query: str) -> str:
    """Perform a web search using DuckDuckGo to find the latest information."""
    return await search_tool.arun(query)

@mcp.tool()
async def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform arithmetic operations: add, sub, mul, div on two numbers."""
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

@mcp.tool()
async def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=VJ5UT2AO4KBQKZJT"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=5)

        if r.status_code != 200:
            return {"error": f"API request failed with status {r.status_code}"}

        data = r.json()
        if "Note" in data:
            return {"error": "API rate limit reached"}

        price = data.get("Global Quote", {}).get("05. price")
        if not price:
            return {"error": "Stock price not available"}

        return {
            "symbol": symbol.upper(),
            "price": float(price)
        }
    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Start the server using the default transport (stdio) 
    mcp.run()
