import requests
import json
import os
from pydantic import BaseModel,Field
from typing import Optional

class GeminiCustomGenerativeAI():
    # model_name: str = Field("custom_llm", description="The name of the custom LLM.")
    # api_key: Optional[str] = Field(None, description="API Key for Google Gemini API")
    # base_url: str = Field("https://generativelanguage.googleapis.com/v1beta/models",
    #                       description="Base URL for Google Gemini API")

    def __init__(self):


        self.api_key = os.environ.get("GEMINI_API_KEY")

        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"


    def generate_content(self,prompt=None, model="gemini-2.0-flash",  temperature=0.7, max_output_tokens=256):

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(headers)
        payload = {
             "contents": [
                  {
                    "parts": [
                      {
                        "text": prompt
                      }
                    ]
                  }
                ]
        }

        try:
            response = requests.post(url,json=payload,verify=False)
            response.raise_for_status()  # Raise an exception for HTTP errors
            result=response.json()
            print(result)
            return result['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_embeddings(self, texts):
        embedding_url = f"{self.base_url}/gemini-embedding-exp-03-07:embedContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": "models/gemini-embedding-exp-03-07",
             "content": {
             "parts":[{
             "text": texts}]}
    }

        response = requests.post(embedding_url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            #print(response_data)
            return response_data['embedding']['values']
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


# Example usage
if __name__ == "__main__":


    # Initialize the CustomGenerativeAI client
    custom_ai = GeminiCustomGenerativeAI()

    # Provide a prompt for content generation
    prompt = "Write a short story about bangalore food"

    # Generate content
    try:
        response = custom_ai.generate_content(prompt="Tell me a joke")
        # model = "gemini-2.0-flash",
        # prompt = prompt,
        # temperature = 0.6,
        # max_output_tokens = 150
        print("Generated Content:")

        print(response)
    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"Error: {e}")