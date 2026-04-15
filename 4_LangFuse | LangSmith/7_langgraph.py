# pip install -U langgraph langchain-groq pydantic python-dotenv langfuse

import operator
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# ---------- Setup ----------
load_dotenv()
langfuse_handler = CallbackHandler()

model = ChatGroq(
    model= 'moonshotai/kimi-k2-instruct-0905',
    temperature= 0
)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# Use json_mode for better reliability with Groq's Llama models
structured_model = model.with_structured_output(EvaluationSchema, method="json_mode")

# ---------- Sample essay ----------
essay2 = """
        India and AI Time

        Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

        India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program "AI for All". It want AI in farm, doctor place, school and transport.

        In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

        But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

        One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

        India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

        If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

        So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
    """

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists
    avg_score: float

# ---------- Observed node functions ----------
@observe(name="evaluate_language_fn")
def evaluate_language(state: UPSCState):
    with propagate_attributes(
        tags=["evaluation", "language"],
        metadata={"dimension": "language", "model": "llama-3.1-8b-instant"}
    ):
        system_msg = (
            "You are an expert UPSC examiner. You must respond with a JSON object "
            "containing 'feedback' (string) and 'score' (integer 0-10). "
            "Do not include any other text or explanation outside the JSON."
        )
        prompt = f"Evaluate the LANGUAGE quality of this essay:\n\n{state['essay']}"
        
        messages = [("system", system_msg), ("human", prompt)]
        out = structured_model.invoke(messages)
        return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@observe(name="evaluate_analysis_fn")
def evaluate_analysis(state: UPSCState):
    with propagate_attributes(
        tags=["evaluation", "analysis"],
        metadata={"dimension": "analysis", "model": "llama-3.1-8b-instant"}
    ):
        system_msg = (
            "You are an expert UPSC examiner. You must respond with a JSON object "
            "containing 'feedback' (string) and 'score' (integer 0-10). "
            "Do not include any other text or explanation outside the JSON."
        )
        prompt = f"Evaluate the DEPTH OF ANALYSIS of this essay:\n\n{state['essay']}"
        
        messages = [("system", system_msg), ("human", prompt)]
        out = structured_model.invoke(messages)
        return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@observe(name="evaluate_thought_fn")
def evaluate_thought(state: UPSCState):
    with propagate_attributes(
        tags=["evaluation", "clarity"],
        metadata={"dimension": "clarityofthought", "model": "llama-3.1-8b-instant"}
    ):
        system_msg = (
            "You are an expert UPSC examiner. You must respond with a JSON object "
            "containing 'feedback' (string) and 'score' (integer 0-10). "
            "Do not include any other text or explanation outside the JSON."
        )
        prompt = f"Evaluate the CLARITY OF THOUGHT of this essay:\n\n{state['essay']}"
        
        messages = [("system", system_msg), ("human", prompt)]
        out = structured_model.invoke(messages)
        return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@observe(name="final_evaluation_fn")
def final_evaluation(state: UPSCState):
    with propagate_attributes(
        tags=["evaluation", "aggregate"],
        metadata={"dimension": "overall", "model": "llama-3.1-8b-instant"}
    ):
        prompt = (
            "Based on the following feedback, create a summarized overall feedback.\n\n"
            f"Language feedback: {state.get('language_feedback','')}\n"
            f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
            f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
        )
        overall = model.invoke(prompt).content
        scores = state.get("individual_scores", []) or []
        avg = (sum(scores) / len(scores)) if scores else 0.0
        return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Invoke ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay",
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essaylength": str(len(essay2)),
                "model": "llama-3.1-8b-instant",
                "dimensions": "language,analysis,clarity",
            },
            "callbacks": [langfuse_handler]   # ← Langfuse tracing for the chain
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))