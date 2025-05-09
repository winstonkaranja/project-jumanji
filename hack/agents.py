import os
from typing import Any
import boto3
from dotenv import load_dotenv
import tifffile as tiff
import numpy as np
from io import BytesIO

from PIL import Image

import matplotlib.pyplot as plt

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

from langchain_core.prompts import ChatPromptTemplate



from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from PIL import Image
import base64
import requests
import os

from compute import full_image_processing_pipeline, process_ndvi_for_carbon_credits, read_image_from_s3



# api_key = getpass.getpass("Enter your OpenAI API Key: ")

# Set the API key as an environment variable
os.environ.clear()
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]



# Load the Image and Convert to Base64
def encode_image_from_s3(bucket_name, key):
    """
    Load an image from S3 (using the shared read_image_from_s3 function) and encode it as a Base64 string.
    This function supports only JPEG images for encoding.
    """
    # Ensure the file is a JPEG based on its S3 key extension.
    key_lower = key.lower()
    if not key_lower.endswith(('.jpg', '.jpeg')):
        raise ValueError("Only JPEG images can be encoded to Base64.")
    
    # Load the image using the shared read_image_from_s3 function.
    image = read_image_from_s3(bucket_name, key)
    
    # Ensure the returned image is a PIL Image instance (as expected for JPEGs).
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a JPEG image (as PIL Image), but received a different type.")
    
    # Save the PIL image to a BytesIO buffer.
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Encode the image bytes to Base64.
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64






from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field



# Define the expected structured output using Pydantic.
class PlantAnalysisOutput(BaseModel):
    plant: str = Field(..., description="The identified plant type, e.g., 'Maize'")
    status: str = Field(..., description="Must be 'approve' if a plant is found")
    carbon_fraction: float = Field(..., description="Carbon fraction value from the CSV")
    root_shoot_ratio: float = Field(..., description="Root-to-shoot ratio value from the CSV")

def analyze_image_and_get_response(bucket_name: str, key: str, llm) -> dict:
    """
    Encodes the image from S3, prepares messages, and calls the LLM (with structured output)
    to analyze the image.

    Args:
        bucket_name (str): The S3 bucket where the image is stored.
        key (str): The key (path) of the image in the S3 bucket.
        llm: An instance of a ChatOpenAI (or similar) LLM.

    Returns:
        dict: The structured response (if a plant is detected) or the string "deny".
    """
    # Encode the image from S3 into a base64 string.
    image_base64 = encode_image_from_s3(bucket_name, key)
    
    # Prepare messages.
    messages = [
        SystemMessage(content="You are an AI that analyzes images."),
        HumanMessage(content="""
            You are an image moderation and analysis system with an integrated plant parameter module.
            Your task is twofold:
            1. Analyze the input image to determine if it clearly depicts a plant or plantation (e.g., crops, greenery, trees, leaves, or agricultural land).
            2. If a plant is detected, identify the most likely plant type from the sample CSV data provided and return the corresponding parameters.
            If the image is not a plant/plantation, simply reply with the single word 'deny' (all lowercase).

            If the image depicts a plant, your response must be a JSON object with the following keys:
            - plant: the plant type (e.g., 'Maize', 'Soybean', etc.)
            - status: the word 'approve'
            - carbon_fraction: the carbon fraction value from the CSV
            - root_shoot_ratio: the root-to-shoot ratio value from the CSV

            Use the following CSV data as your reference:

            Plant,Carbon Fraction,Root-Shoot Ratio
            Maize,0.45,0.20
            Soybean,0.50,0.15
            Wheat,0.47,0.18
            Rice,0.46,0.16

            Ensure your output strictly follows this JSON structure if a plant is detected. Otherwise, output only the word 'deny'.
                    """),
        HumanMessage(content=[{
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        }])
    ]
    
    # Wrap the LLM with structured output enforcement.
    structured_llm = llm.with_structured_output(PlantAnalysisOutput)
    
    try:
        # Invoke the LLM with the structured output parser.
        result = structured_llm.invoke(messages)
        return result  # This will be a dict matching PlantAnalysisOutput.
    except Exception as e:
        # If structured parsing fails (for example, if the output is "deny"),
        # fallback to a raw LLM invocation.
        raw_output = llm.invoke(messages)
        if raw_output.strip() == "deny":
            return "deny"
        raise ValueError(f"Unexpected LLM output: {raw_output}") from e






# -------------------------------------------------------
# Step 2: Wrap NDVI Check as a LangChain Tool for Aerial Photo
# -------------------------------------------------------



# -------------------------------
# Global Parameters for the Pipeline
# -------------------------------
RADIOMETRIC_PARAMS = {
    'gain': [0.012, 0.012, 0.012, 0.012, 0.012],
    'offset': [0, 0, 0, 0, 0],
    'sunelev': 60.0,
    'edist': 1.0,
    'Esun': [1913, 1822, 1557, 1317, 1074],
    'blackadjust': 0.01,
    'low_percentile': 1
}
NOISE_METHOD = 'median'
NOISE_KERNEL_SIZE = 3
SIGMA = 1.0



# -------------------------------------------------------
# Step 2: Wrap NDVI Check as a LangChain Tool for Aerial Photo
# -------------------------------------------------------
@tool
def check_ndvi(
    # Example S3 file reading (ensure read_tiff_from_s3 is defined and imported)
    key: str, 
    bucket_name: str = "qijaniproductsbucket",
    red_band_index: int = 2, 
    nir_band_index: int = 4,
    radiometric_params: dict = RADIOMETRIC_PARAMS,
    noise_method: str = NOISE_METHOD,
    noise_kernel_size: int = NOISE_KERNEL_SIZE,
    sigma: float = SIGMA,
    save_path: str = "ndvi_output.jpg"
) -> tuple[np.ndarray, str]:
    """
    Given an aerial multispectral image (as a nested list) with pixel values in [0,1],
    compute the NDVI using the provided red and NIR band indices.
    Returns:
    - Processed NDVI array (NumPy array)
    - Path to the saved NDVI image file
    """

    image = read_image_from_s3(bucket_name, key)

    # Run NDVI processing pipeline
    ndvi_noise_reduced, _= full_image_processing_pipeline(
        image,
        radiometric_params, 
        detector_type='ORB',
        noise_method=noise_method, 
        noise_kernel_size=noise_kernel_size, 
        sigma=sigma,
        nir_band_index=nir_band_index, 
        red_band_index=red_band_index, 
        visualize=False,
        use_parallel_noise_reduction=False
    )

    # Plot and save the NDVI image
    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi_noise_reduced, cmap='RdYlGn')
    plt.colorbar(label='NDVI Value')
    plt.title("NDVI Analysis")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # Stacking along new axis
    ndvi_noise_reduced = np.concatenate(ndvi_noise_reduced)

    # Save NDVI data to a file (e.g., using NumPy's save)
    np.save("ndvi_result2.npy", ndvi_noise_reduced)
    return ndvi_noise_reduced




def aerial_photo_analysis(key: str, llm) -> dict:
    """
    Runs the aerial photo analysis pipeline using the check_ndvi tool.
    
    Args:
        llm: The language model instance (e.g., ChatOpenAI).
        key (str): The image key from S3 to be processed.
        
    Returns:
        dict: The result returned by the agent.
    """
    # Define the prompt for the agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are an agricultural image evaluator. Given the following multispectral aerial image key, "
                "use the check_ndvi tool to compute NDVI. Determine if the NDVI figure produced in the output file "
                "save path indicates real farmland. Then, output either 'approve' or 'deny' (short and precise)."
            )),
            ("human", "Image: " + key),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    
    # Include the NDVI checking tool in our list of tools (assumes check_ndvi is defined)
    tools = [check_ndvi]
    
    # Initialize the tool-calling agent using the provided LLM, tools, and prompt
    agent = create_tool_calling_agent(llm, tools, prompt)
    validity_agent = AgentExecutor(agent=agent, tools=tools)
    
    # Run the agent (the "input" field is used as the initial message for the agent)
    result = validity_agent.invoke({"input": "Image: " + key})
    return result

