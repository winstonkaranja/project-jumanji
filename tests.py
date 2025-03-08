from langchain_openai import ChatOpenAI

from agents import analyze_image_and_get_response

# Instantiate the LLM (ensure your API key and other settings are configured)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define your S3 bucket name and image key
bucket_name = "qijaniproductsbucket"
key = "Maize.jpg"

# Call the function to analyze the image
response = analyze_image_and_get_response(bucket_name, key, llm)

# Print the LLM response
print("LLM Response:", response)
