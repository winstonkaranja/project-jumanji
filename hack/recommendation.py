from langchain_anthropic import ChatAnthropic

from typing import NotRequired, TypedDict

# Import required LangGraph components
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import numpy as np


from agents import PlantAnalysisOutput, aerial_photo_analysis, analyze_image_and_get_response



import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["ANTHROPIC_API_KEY"]

llm = ChatAnthropic(model_name="claude-3-5-haiku-latest", api_key=api_key)

response = llm.invoke(
    "Write a haiku about how to use LangChain with AWS services"
)    
print(response)


# Define the state schema
class State(TypedDict):
    user_id: NotRequired[str]  # to store the user ID from the input str
    ground_key: str
    aerial_key: PlantAnalysisOutput
    analyze_result: NotRequired[str]  # to store the LLM response from image analysis
    aerial_result: NotRequired[str]   # to store the result from aerial photo analysis
    carbon_credits: NotRequired[float]



def node_analyze_rgb(state: State) -> dict:

    # Check if user_id is present in the state
    if "user_id" not in state:
        raise ValueError("user_id not found in state")

    # Check if ground_key is present in the state
    if "ground_key" not in state:
        raise ValueError("ground_key not found in state")

    # Call the first function using bucket_name and key from the state and the global llm
    result = analyze_image_and_get_response("qijaniproductsbucket", state["ground_key"], llm)
    state["analyze_result"] = result

    return {"user_id": state["user_id"],"analyze_result": state["analyze_result"]}




# Define the node that calls aerial_photo_analysis
def node_aerial(state: State) -> dict:
    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Check if user_id is present in the state
    if "user_id" not in state:
        raise ValueError("user_id not found in state")

    # Check if aerial_key is present in the state
    if "aerial_key" not in state:
        raise ValueError("aerial_key not found in state")

    # Call the aerial photo analysis using the key and global llm
    result = aerial_photo_analysis(state["aerial_key"], llm)
    state["aerial_result"] = result

    #Compute credits for accepted images
    if "approve" in str(result).lower():
            
            print("Image accepted, computing credits...")

            ndvi_noise_reduced = np.load("ndvi_result2.npy")

            default_carbon_fraction = 0.47

            carbon_fraction = default_carbon_fraction
            root_shoot_ratio = None  # Optional: default/fallback value

            try:
               status = state["analyze_result"].status
               carbon_fraction = state["analyze_result"].carbon_fraction
               root_shoot_ratio = state["analyze_result"].root_shoot_ratio

            except Exception:
                pass  # fallback to default

            if status == "approve":
                print("Image approved, computing credits...")
                credits = process_ndvi_for_carbon_credits(ndvi_noise_reduced, carbon_fraction=carbon_fraction, pixel_area_m2=0.01, root_shoot_ratio=root_shoot_ratio)
                state["carbon_credits"] = credits  # Store the credits in the state

            print("Credits computed:", credits)
    else:
            raise Exception("Image rejected!")

    return { "user_id": state["user_id"],"aerial_result": state["aerial_result"], "carbon_credits": state["carbon_credits"]} 