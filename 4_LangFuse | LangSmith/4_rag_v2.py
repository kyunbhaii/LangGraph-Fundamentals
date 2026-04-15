from langfuse import observe, propagate_attributes
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
@observe(name='load_pdf')
def load_pdf(path: str):
    with propagate_attributes(
        tags=['pdf_loader', 'ingestion'],
        metadata={'loader': 'PyPDFLoader', 'format': 'pdf'}
    ):
        laoder = PyPDFLoader(path)
        docs = laoder.load()
        return docs

# 2. Chunking
@observe(name='chunk documents')
def chunk_docs(docs, chunk_size = 1000, chunk_overlap = 150):
    with propagate_attributes(
        tags=['chunking', 'text_splitting'],
        metadata={'chunksize': str(chunk_size), 'chunkoverlap': str(chunk_overlap), 'splitter': 'RecursiveCharacterTextSplitter'}
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size, chunk_overlap = chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return chunks

# Embed + Index
@observe(name='build_vectorstore')
def embed_chunks(chunks):
    with propagate_attributes(
        tags=['embedding', 'faiss', 'vectorstore'],
        metadata={'model': 'all-MiniLM-L6-v2', 'provider': 'huggingface', 'vectorstore': 'FAISS'}
    ):
        emb = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        vs = FAISS.from_documents(chunks, emb)
        return vs

# Pipeline
@observe(name='setup_pipeline')
def setup_pipeline(pdf_path: str):
    with propagate_attributes(
        tags=['rag', 'pipeline', 'setup'],
        metadata={'searchtype': 'similarity', 'topk': '4', 'version': 'v2'}
    ):
        docs = load_pdf(pdf_path)
        chunks = chunk_docs(docs)
        vs = embed_chunks(chunks)
        retriever = vs.as_retriever(search_type='similarity', search_kwargs={'k': 4})
        return retriever
    

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

# Build pipeline
retriever = setup_pipeline(PDF_PATH)

parallel = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

CONFIG = {
    'run_name': 'RAG_v2: Obeservable',
    'tags': ['llm_app', 'report_generation', 'Obeservable', 'Traceable Everywhere'],
    'metadata': {'model': 'llama-3.1-8b-instant', 'model_temp': 0.7, 'speciality': 'Obeservable'},
    'callbacks': [langfuse_handler]
}

# Ask Questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip(), config = CONFIG)
print("\nA:", ans)

"""
chain.invoke("How many diseases?")
        │
        ▼
   RunnableParallel
   ┌─────────────────────────────────────────────────┐
   │                                                 │
   │  'context': retriever | format_docs             │
   │      │                                          │
   │      ▼                                          │
   │  retriever.invoke("How many diseases?")         │
   │      │                                          │
   │      ├─ 1. Embeds the query using the SAME      │
   │      │     HuggingFace embedding model           │
   │      │     (all-MiniLM-L6-v2) automatically     │
   │      │                                          │
   │      ├─ 2. Runs FAISS similarity search         │
   │      │     against stored chunk embeddings       │
   │      │                                          │
   │      └─ 3. Returns top-k documents              │
   │             │                                   │
   │             ▼                                   │
   │         format_docs → joined text string        │
   │                                                 │
   │  'question': RunnablePassthrough()              │
   │      │                                          │
   │      └─ Passes "How many diseases?" as-is       │
   │                                                 │
   └─────────────────────────────────────────────────┘
        │
        ▼
   prompt gets {context: "...", question: "How many diseases?"}

"""