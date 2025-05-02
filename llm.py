import abc
import logging
import requests
import os
import json


# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseLLM(abc.ABC):
    """Abstract base class for LLMs."""
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt."""
        pass

class OllamaLLM(BaseLLM):
    """Concrete LLM using Ollama's API for LLaMA 3 8B."""
    def __init__(self):
        logging.info("Initializing OllamaLLM with LLaMA 3 8B")
        self.model_name = "llama3:8b"
       
        ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
        self.api_url = f"http://{ollama_host}/api/generate"
        logging.info(f"Ollama API URL set to: {self.api_url}")
    
    def generate(self, prompt: str) -> str:
        try:
            test_response = requests.get(self.api_url.replace("/generate", "/tags"))
            if test_response.status_code != 200:
                raise Exception(f"Ollama server not reachable: {test_response.status_code} - {test_response.text}")
            
            PROMPT = f"{prompt}\nProvide a to the point brief answer to the asked question using the context."
           
            payload = {
                "model": self.model_name,
                "prompt": PROMPT,
                # "temperature": 0.5,
                # "top_p": 0.85,
                "max_tokens": 150, 
                "stream": True
            }

            with requests.post(self.api_url, json=payload, stream=True) as response:
                response.raise_for_status()  # Raise an error for bad status codes
                full_response = ""
                for line in response.iter_lines():
                    if line:  # Skip empty lines
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "response" in chunk:
                                full_response += chunk["response"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error in chunk: {str(e)} - Chunk: {line.decode('utf-8')}")
                            continue
                if not full_response:
                    return "Error: No response from Ollama"
                # Return the full response without incorrect "Answer:" extraction
                return full_response.strip()
        except requests.exceptions.ConnectionError:
            logging.error(f"Failed to connect to Ollama server at {self.api_url}. Ensure the server is running.")
            return "Error: Ollama server not running or unreachable."
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error from Ollama: {str(e)}")
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            logging.error(f"Error generating response with Ollama: {str(e)}")
            return f"Error: {str(e)}"