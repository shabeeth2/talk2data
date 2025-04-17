from google.genai import types
import os
from google import genai
google_api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=google_api_key)
# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
from google import genai
model_id = "gemini-1.5-flash"

client = genai.Client(api_key=google_api_key)
prompt = """
    What is the sum of the first 50 prime numbers?
    Generate and run code for the calculation, and make sure you get all 50.
"""

response = client.models.generate_content(
    model=model_id,
    contents=prompt,
    config = types.GenerateContentConfig(
        tools=[types.Tool(
            code_execution=types.ToolCodeExecution
            )]
        )
    )

# print(response.code_execution_result)
def get_code_response(response):
    if response.code_execution_result:
        return response.code_execution_result
    else:
        return None