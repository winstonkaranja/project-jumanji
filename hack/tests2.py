from langchain_openai import ChatOpenAI


from hack.agents import aerial_photo_analysis

# Instantiate your LLM (ensure your API key is configured appropriately)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Provide the S3 image key to be analyzed
image_key = "20180627_seq_50m_NC.tif"

# Call the aerial photo analysis pipeline function
result = aerial_photo_analysis(llm, image_key)

# Print the final agent decision
print("Final agent decision:", result)
