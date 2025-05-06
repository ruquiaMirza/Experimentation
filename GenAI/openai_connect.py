import requests
import json
import os

class OpenAICustomGenerativeAI:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"
        self.EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

    def generate_content(self, prompt, model="gpt-3.5-turbo", max_tokens=100, temperature=0.7):

        # Define the headers for the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Define the payload for the request
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        })

        try:
            # Make the POST request to the OpenAI API
            response = requests.post(self.url, headers=headers, data=payload)

            # Raise an error if the request was unsuccessful
            response.raise_for_status()

            # Parse the response JSON
            response_data = response.json()

            # Extract and return the generated text
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"
        except KeyError:
            return "Unexpected response format from OpenAI API."

    # Or "text-embedding-3-large", "text-embedding-ada-002"
    def get_embeddings(self,texts,model= "text-embedding-3-small", encoding_format= "float", max_retries: int = 3, ** kwargs):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if not texts:
            raise ValueError("The input 'texts' list cannot be empty.")

        if encoding_format not in ["float", "base64", "int"]:
            raise ValueError(f"Invalid encoding_format: {encoding_format}.  Must be 'float', 'base64', or 'int'.")

        payload = {
            "model": model,
            "input": texts,
            "encoding_format": encoding_format,
            **kwargs  # Include any additional parameters
        }
        json_payload = json.dumps(payload)

        for attempt in range(max_retries):
            try:
                response = requests.post(self.EMBEDDINGS_URL, headers=headers, data=json_payload)
                response.raise_for_status()  # Raise HTTPError for bad status codes
                data = response.json()

                # Extract embeddings from the response
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings

            except requests.exceptions.RequestException as e:
                print(f"Error during OpenAI API request (attempt {attempt + 1}/{max_retries}): {e}")
                if response is not None:
                    print(f"Response status code: {response.status_code}")
                    print(f"Response body: {response.text}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                else:
                    raise RuntimeError(f"Failed to get embeddings after {max_retries} attempts.  Last error: {e}")



if __name__ == "__main__":
    # Example usage

    prompt = "Write a short poem about the beauty of nature."

    openaiConnect=OpenAICustomGenerativeAI()
    try:
        generated_text = openaiConnect.generate_content(prompt)
        print("Generated Content:")
        print(generated_text)
    except Exception as e:
        print(f"Error: {e}")

    texts_to_embed = [
        "This is the first text to embed.",
        "This is the second text.",
        "A third text for embedding.",
    ]

    try:
        float_embeddings = openaiConnect.get_embeddings(texts=texts_to_embed, model="text-embedding-3-small")
        print(
            f"Float Embeddings (first embedding):\n{float_embeddings[0][:5]}...\n")  # Print only the first 5 elements for brevity
    except Exception as e:
        print(f"An error occurred: {e}")