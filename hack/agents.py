from typing import Any
import boto3
from dotenv import load_dotenv
import tifffile as tiff
import numpy as np
from io import BytesIO


import os
from PIL import Image

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
import openai



from langchain.schema import HumanMessage
from langchain.schema.messages import SystemMessage
from PIL import Image
import base64
import requests
import os

from compute import full_image_processing_pipeline, process_ndvi_for_carbon_credits







def read_image_from_s3(bucket_name, key):
    """
    Download an image from S3 and load it based on its file type.
    Supports TIFF (.tif, .tiff) and JPEG (.jpg, .jpeg) formats.
    
    For TIFF images, the image is returned as a NumPy array (float32).
    For JPEG images, the image is returned as a PIL.Image instance.
    """
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
        region_name=os.environ.get('AWS_REGION')
    )

    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()
    
    # Create a file-like object from the data
    file_obj = BytesIO(image_data)
    
    # Check file extension to decide how to load the image
    key_lower = key.lower()
    if key_lower.endswith(('.tif', '.tiff')):
        # Load TIFF image using tifffile and convert to float32 NumPy array
        image = tiff.imread(file_obj).astype(np.float32)
    elif key_lower.endswith(('.jpg', '.jpeg')):
        # Load JPEG image using PIL
        image = Image.open(file_obj)
    else:
        raise ValueError("Unsupported image format. Please use TIFF or JPEG.")
    
    return image




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






def analyze_image_and_get_response(bucket_name: str, key: str, llm) -> dict:
    """
    Encodes the image from S3, prepares messages, and calls the LLM to analyze the image.

    Args:
        bucket_name (str): The S3 bucket where the image is stored.
        key (str): The key (path) of the image in the S3 bucket.
        llm: An instance of a ChatOpenAI (or similar) LLM.

    Returns:
        dict: The response returned by the LLM.
    """
    # Encode the image from S3 into a base64 string.
    image_base64 = encode_image_from_s3(bucket_name, key)
    
    # Prepare the messages with both text and the image URL (embedded in base64).
    messages = [
        SystemMessage(content="You are an AI that analyzes images."),
        HumanMessage(content=[
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ]
    
    # Invoke the LLM with the prepared messages.
    response = llm.invoke(messages)
    return response








# -------------------------------------------------------
# Step 2: Wrap NDVI Check as a LangChain Tool for Aerial Photo
# -------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt

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
                "save path indicates real farmland. Then, output either 'Accept' or 'Reject' (short and precise)."
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
    #Compute credits for accepted images
    if "accept" in str(result).lower():
            def compute_credits(ndvi_noise_reduced, pixel_area_m2=0.01):
                """Assuming process_ndvi_for_carbon_credits accepts a list of NDVI values."""
                credits = process_ndvi_for_carbon_credits(ndvi_noise_reduced, pixel_area_m2)
                return credits

            # Usage: assuming ndvi_noise_reduced is defined elsewhere
            ndvi_noise_reduced = np.load("ndvi_result2.npy")
            credits = compute_credits(ndvi_noise_reduced)
            print("Credits computed:", credits)
    else:
            raise Exception("Image rejected")

    return result




