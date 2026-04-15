from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

PDF_PATH = 'Policy_HDFC optima.pdf'

# 1. Load PDF
laoder = PyPDFLoader(PDF_PATH)
docs = laoder.load()

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
chunks = splitter.split_documents(docs)

# Embed + Index
emb = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vs = FAISS.from_documents(chunks, emb)
retriever = vs.as_retriever(search_type = 'similarity', search_kwargs = {'k':4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# Chain
llm = ChatGroq(
    model= 'llama-3.1-8b-instant',
    temperature= 0.7
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

CONFIG = {
    'run_name': 'RAG_v1',
    'tags': ['llm_app', 'report_generation', 'sequential_chain'],
    'metadata': {'model': 'llama-3.1-8b-instant', 'model_temp': 0.7, 'parser': 'StrOutputParser'},
    'callbacks': [langfuse_handler]
}

# Ask Questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip(), config = CONFIG)
print("\nA:", ans)