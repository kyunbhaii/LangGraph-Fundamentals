# LangGraph Fundamentals

A hands-on guide to LangGraph covering state graphs, persistence, chatbot memory, human-in-the-loop (HITL), tools, subgraphs, and agentic RAG pipelines (CRAG & Self-RAG) with practical Python examples.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## Topics Covered

| # | Module | What You'll Learn |
|---|--------|-------------------|
| 1 | [Workflows](#1-workflows) | State graphs, nodes, edges, compiled graphs |
| 2 | [Persistence](#2-persistence) | Checkpointers, state saving, SQLite integration |
| 3 | [Chatbot](#3-chatbot) | Building stateful conversational agents |
| 4 | [LangFuse & LangSmith](#4-observability-langfuse--langsmith) | Agentic observability, tracing, debugging |
| 5 | [Tools](#5-tools) | Tool calling, agent executors, tool nodes |
| 6 | [HITL (Human in the Loop)](#6-hitl-human-in-the-loop) | Pausing execution, manual approval, state updates |
| 7 | [Subgraphs](#7-subgraphs) | Modular graphs, parent/child graph communication |
| 8 | [Memory](#8-memory) | Advanced memory management in agents |
| 9 | [CRAG](#9-crag-corrective-rag) | Corrective RAG, relevance grading, fallback web search |
| 10 | [Self-RAG](#10-self-rag) | Agentic RAG, self-correction loops, hallucination grading |

---

## Project Structure

```
langgraph-fundamentals/
│
├── 1_Workflows/
├── 2_Persistence/
├── 3_Chatbot/
├── 4_LangFuse | LangSmith/
├── 5_Tools/
├── 6_HITL/
├── 7_Subgraphs/
├── 8_Memory/
├── 9_CRAG/
├── 10_Self_RAG/
└── Dataset/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for the models you want to use (e.g., Groq, OpenAI, Tavily)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kyunbhaii/LangGraph-Fundamentals.git
   cd LangGraph-Fundamentals
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   ```

---

## Module Breakdown

### 1. Workflows
- Introduction to `StateGraph`, `START`, and `END` nodes.
- Defining state schemas using `TypedDict` and `Pydantic`.
- Creating nodes and routing via conditional edges.

### 2. Persistence
- Using checkpointers (e.g., `MemorySaver`, `SqliteSaver`) to persist graph state.
- Resuming workflows from previous states and exploring thread memory.

### 3. Chatbot
- Building a stateful chatbot with persistent chat history.
- Handling multi-turn conversations through graph nodes.

### 4. Observability (LangFuse & LangSmith)
- Integrating observability tools to trace node execution, token usage, and latency.

### 5. Tools
- Giving agents the ability to call external tools.
- Using `ToolNode` and conditional routing for tool execution.

### 6. HITL (Human-in-the-Loop)
- Pausing graphs at specific nodes using `interrupt_before`.
- Manual approval workflows and modifying state values dynamically.

### 7. Subgraphs
- Breaking down complex agents into manageable subgraphs.
- Passing states between parent graphs and child subgraphs.

### 8. Memory
- Advanced memory management across multi-agent workflows.

### 9. CRAG (Corrective RAG)
- Implementing relevance grading on retrieved documents.
- Query rewriting and using Tavily search as a fallback when documents are irrelevant.

### 10. Self-RAG
- Complex agentic RAG with self-correction loops.
- Grading answer relevance, grounding (IsSUP), and usefulness (IsUSE).
- Strict JSON schema parsing with Groq API for deterministic agent routing.

---

## Contributing

This is a learning repository. Feel free to fork it, open issues, or submit PRs to improve examples or add new topics.

---

## Acknowledgments

This codebase was built while learning from the excellent **[LangGraph tutorial series by CampusX](https://youtube.com/playlist?list=PLKnIA16_RmvYsvB8qkUQuJmJNuiCUJFPL&si=F3Deu-QWo5Xu5mZY)** on YouTube. Special thanks to Nitish Sir (CampusX) for the clear and structured explanations!

---

## License

This project is open source and available under the MIT License.