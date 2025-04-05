from typing import NotRequired, TypedDict

# Import required LangGraph components
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from regex import D

from agents import aerial_photo_analysis, analyze_image_and_get_response

# Define the state schema
class State(TypedDict):
    bucket_name: str
    ground_key: str
    aerial_key: str
    analyze_result: NotRequired[dict]  # to store the LLM response from image analysis
    aerial_result: NotRequired[dict]   # to store the result from aerial photo analysis




# Define the node that calls analyze_image_and_get_response
def node_analyze(state: State) -> dict:
    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Call the first function using bucket_name and key from the state and the global llm
    result = analyze_image_and_get_response(state["bucket_name"], state["ground_key"], llm)
    return {"analyze_result": result}

# Define the node that calls aerial_photo_analysis
def node_aerial(state: State) -> dict:
    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Call the aerial photo analysis using the key and global llm
    result = aerial_photo_analysis(state["aerial_key"], llm)
    return {"aerial_result": result}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("analyze", node_analyze)
graph_builder.add_node("aerial", node_aerial)

# Set up edges:
# 1. From START to the "analyze" node.
# 2. From "analyze" node to "aerial" node.
# 3. From "aerial" node to END.
graph_builder.add_edge(START, "analyze")
graph_builder.add_edge("analyze", "aerial")
graph_builder.add_edge("aerial", END)

# Compile the graph into a runnable object
compiled_graph = graph_builder.compile()

# Now compiled_graph is ready to be used (for example, via compiled_graph.invoke(state))
