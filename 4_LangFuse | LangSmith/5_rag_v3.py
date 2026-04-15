# pip install -U langchain langchain-groq langchain-community langchain-huggingface faiss-cpu pypdf python-dotenv langfuse

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

langfuse_handler = CallbackHandler()

PDF_PATH = "Policy_HDFC optima.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- helpers (traced) -----------------
@observe(name="load_pdf")
def load_pdf(path: str):
    with propagate_attributes(
        tags=["pdf_loader", "ingestion"],
        metadata={"loader": "PyPDFLoader", "format": "pdf"}
    ):
        return PyPDFLoader(path).load()

@observe(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    with propagate_attributes(
        tags=["chunking", "text_splitting"],
        metadata={"chunksize": str(chunk_size), "chunkoverlap": str(chunk_overlap), "splitter": "RecursiveCharacterTextSplitter"}
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return chunks

@observe(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    with propagate_attributes(
        tags=["embedding", "faiss", "vectorstore"],
        metadata={"model": embed_model_name, "provider": "huggingface", "vectorstore": "FAISS"}
    ):
        emb = HuggingFaceEndpointEmbeddings(model=embed_model_name)
        vs = FAISS.from_documents(splits, emb)
        return vs

# ----------------- cache key / fingerprint -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
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

# ----------------- explicitly traced load/build runs -----------------
@observe(name="load_index")
def load_index_run(index_dir: Path, embed_model_name: str):
    with propagate_attributes(
        tags=["index", "cache_hit"],
        metadata={"indexdir": str(index_dir), "model": embed_model_name}
    ):
        emb = HuggingFaceEndpointEmbeddings(model=embed_model_name)
        return FAISS.load_local(
            str(index_dir),
            emb,
            allow_dangerous_deserialization=True
        )

@observe(name="build_index")
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    with propagate_attributes(
        tags=["index", "cache_miss"],
        metadata={"pdfpath": os.path.basename(pdf_path), "chunksize": str(chunk_size)}
    ):
        docs = load_pdf(pdf_path)
        splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vs = build_vectorstore(splits, embed_model_name)
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
        (index_dir / "meta.json").write_text(json.dumps({
            "pdf_path": os.path.abspath(pdf_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embed_model_name,
        }, indent=2))
        return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = EMBED_MODEL,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------- model, prompt, and pipeline -----------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@observe(name="setup_pipeline")
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150, embed_model_name=EMBED_MODEL, force_rebuild=False):
    with propagate_attributes(
        tags=["rag", "pipeline", "setup"],
        metadata={"version": "v3", "searchtype": "similarity", "topk": "4"}
    ):
        return load_or_build_index(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model_name=embed_model_name,
            force_rebuild=force_rebuild,
        )

@observe(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = EMBED_MODEL,
    force_rebuild: bool = False,
):
    with propagate_attributes(
        tags=["rag", "qa", "full_run"],
        metadata={"version": "v3", "model": "llama-3.1-8b-instant"}
    ):
        vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        chain = parallel | prompt | llm | StrOutputParser()

        CONFIG = {
            "run_name": "RAG_v3: Observable + Cached",
            "tags": ["llm_app", "rag", "cached_index"],
            "metadata": {"model": "llama-3.1-8b-instant", "version": "v3"},
            "callbacks": [langfuse_handler]
        }

        return chain.invoke(question, config=CONFIG)

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG v3 ready (with index caching). Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)