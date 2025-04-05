from typing import NotRequired, TypedDict

# Import required LangGraph components
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import numpy as np
from regex import D

from agents import aerial_photo_analysis, analyze_image_and_get_response
from compute import process_ndvi_for_carbon_credits

# Define the state schema
class State(TypedDict):
    user_id: NotRequired[str]  # to store the user ID from the input str
    ground_key: str
    aerial_key: str
    analyze_result: NotRequired[str]  # to store the LLM response from image analysis
    aerial_result: NotRequired[str]   # to store the result from aerial photo analysis
    carbon_credits: NotRequired[float]




# Define the node that calls analyze_image_and_get_response
def node_analyze(state: State) -> dict:

    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Call the first function using bucket_name and key from the state and the global llm
    result = analyze_image_and_get_response("qijaniproductsbucket", state["ground_key"], llm)
    state["analyze_result"] = result

    return {"analyze_result": result}

# Define the node that calls aerial_photo_analysis
def node_aerial(state: State) -> dict:
    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Call the aerial photo analysis using the key and global llm
    result = aerial_photo_analysis(state["aerial_key"], llm)
    state["aerial_result"] = result

    #Compute credits for accepted images
    if "approve" in str(result).lower():
            
            print("Image accepted, computing credits...")

            ndvi_noise_reduced = np.load("ndvi_result2.npy")
            credits = process_ndvi_for_carbon_credits(ndvi_noise_reduced, pixel_area_m2=0.01)
            state["carbon_credits"] = credits  # Store the credits in the state

            print("Credits computed:", credits)
    else:
            raise Exception("Image rejected")

    return {"aerial_result": state["aerial_result"], "carbon_credits": state["carbon_credits"]} 


# Define the conditional logic function
def analyze_conditional_edge(state: State) -> str:
    result = state["analyze_result"]

    # If result is an AIMessage, extract the text content
    if hasattr(result, "content"):
        result_text = result.content.strip().lower()

    elif isinstance(result, str):
        result_text = result.strip().lower()

    else:
        raise ValueError(f"Unexpected result type: {type(result)}")

    return "aerial" if result_text == "approve" else END



# Build the graph with conditional logic
graph_builder = StateGraph(State)
graph_builder.add_node("analyze", node_analyze)
graph_builder.add_node("aerial", node_aerial)


# Set edges with conditional branching
graph_builder.add_edge(START, "analyze")
graph_builder.add_conditional_edges("analyze", analyze_conditional_edge)
graph_builder.add_edge("aerial", END)

# Compile the graph
compiled_graph = graph_builder.compile()

