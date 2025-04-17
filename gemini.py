from google import genai
import os
google_api_key = os.getenv("GOOGLE_API_KEY")




client = genai.Client(api_key=google_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)
t=client.models.generate_content(,model=model_id,contents=)
print(response.con)