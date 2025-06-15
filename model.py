import openai
import os
import time
import random
from dotenv import load_dotenv

# Load API Key securely
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

if not openai_api_key:
    raise ValueError("API Key is missing! Set it in the .env file.")

client = openai.OpenAI(api_key=openai_api_key)

# Function to query ChatGPT with strict anti-NLP and dermatology restriction
def chat_without_nlp(prompt):
    system_prompt = (
        "You are a strict dermatology assistant. "
        "You will only respond to dermatology-related queries that are correctly formatted and free of typos. "
        #"If the input contains typos, slang, or is unclear, you must respond with: 'Invalid input. Please provide a well-formed dermatology-related query.' "
        "Do not interpret or correct user mistakes. "
        #"Do not guess the intended meaning of typos. "
        #"Do not apply any natural language understanding techniques. "
        "If the user explicitly asks for product recommendations, diet plans, or practices to avoid, then provide that information. "
        "Otherwise, simply answer the dermatology query without listing those three points."
    )

    # Introduce a delay between 3 to 5 minutes (optional)
    #wait_time = random.randint(780, 900)  # Random delay in seconds
    # print(f"Processing... (Response will be delayed by {wait_time // 60} minutes)")
    #time.sleep(wait_time)  # Wait before making API call

    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content

# Example Usage
user_query = "Can you suggest a diet plan for ezcema?"
response = chat_without_nlp(user_query)
print(response)