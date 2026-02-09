import ast
import os
from typing import Any, Dict, List, Optional

import datasets
import yaml
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from tools import (
    add,
    divide,
    get_weather_info,
    read_excel_file,
    web_search,
    wiki_search,
)

load_dotenv()
# =============================================================================
# Configuration
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

with open("system_prompt.yaml", "r") as file:
    SYSTEM_PROMPT = yaml.safe_load(file)["SYSTEM_PROMPT"]

# System message
sys_msg = SystemMessage(content=SYSTEM_PROMPT)

# =============================================================================
# LLM & Tools Setup
# =============================================================================

llm = ChatGroq(model="qwen/qwen3-32b", api_key=GROQ_API_KEY, temperature=0)

tools = [web_search, wiki_search, read_excel_file, get_weather_info, add, divide]
chat_with_tools = llm.bind_tools(tools)

# =============================================================================
# BM25 Retriever
# =============================================================================


def load_bm25_retriever() -> BM25Retriever:
    """Load the dataset and create a BM25 retriever."""
    question_dataset = datasets.load_dataset("ExtarLearn/questions", split="train")
    docs = []

    for data in question_dataset:
        try:
            metadata_dict = ast.literal_eval(data["metadata"])
            doc = Document(page_content=data["content"], metadata=metadata_dict)
            docs.append(doc)
        except Exception as e:
            print(f"Skipping entry due to error: {e}")

    return BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """Retrieves similar questions and answers."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join(
            [f"{i+1}. {doc.page_content}" for i, doc in enumerate(results[:3])]
        )
    return "No matching questions found."


bm25_retriever = load_bm25_retriever()

# =============================================================================
# Agent State & Nodes
# =============================================================================


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task_id: str


def assistant(state: AgentState):
    return {"messages": [chat_with_tools.invoke(state["messages"])]}


def retriever(state: AgentState):
    """Retriever node using BM25 question retriever."""
    user_message = state["messages"][-1].content
    retrieved_info = extract_text(user_message)
    example_msg = HumanMessage(
        content=f"Here are some similar questions and answers for reference:\n\n{retrieved_info}"
    )
    return {"messages": [sys_msg] + state["messages"] + [example_msg]}


# =============================================================================
# Graph Construction
# =============================================================================


def build_graph() -> StateGraph:
    """Build and compile the agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Show the butler's thought process
    # display(Image(builder.get_graph(xray=True).draw_mermaid_png()))

    return builder.compile()
