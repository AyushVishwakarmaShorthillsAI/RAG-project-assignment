import abc
import logging
import requests
import os
import json

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
        self.api_url = f"http://{os.getenv('OLLAMA_HOST', '127.0.0.1:11434')}/api/generate"
        logging.info(f"Ollama API URL set to: {self.api_url}")
    
    def generate(self, prompt: str) -> str:
        try:
            # Test server availability
            test_response = requests.get(self.api_url.replace("/generate", "/tags"))
            if test_response.status_code != 200:
                raise Exception(f"Ollama server not reachable: {test_response.status_code} - {test_response.text}")
            
            prompt_with_instruction = f"{prompt}\nAnswer the question briefly and to the point using the context. Expand only if necessary."
            payload = {
                "model": self.model_name,
                "prompt": prompt_with_instruction,
                "max_tokens": 150,
                "stream": True
            }

            full_response = ""
            with requests.post(self.api_url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
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
            return full_response.strip()
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            logging.error(f"Failed to connect to Ollama server: {str(e)}")
            return "Error: Ollama server not running or unreachable."
        except Exception as e:
            logging.error(f"Error generating response with Ollama: {str(e)}")
            return f"Error: {str(e)}"