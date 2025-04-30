import abc
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(filename='rag_project.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseLLM(abc.ABC):
    """Abstract base class for LLMs."""
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the prompt."""
        pass

class HuggingFaceLLM(BaseLLM):
    """Concrete LLM using HuggingFace transformers."""
    def __init__(self):
        logging.info("Initializing HuggingFaceLLM")
        model_name = "EleutherAI/gpt-neo-1.3B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Device set to use: {'cuda' if device == 0 else 'cpu'}")
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_new_tokens=150,
            temperature=0.6,  # Increased for better diversity
            top_p=0.9,
            do_sample=True
        )
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.generator(
                prompt,
                max_new_tokens=150,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            generated_text = response[0]['generated_text']
            # Extract only the answer part, removing the prompt
            answer_start = generated_text.find("Answer:") + len("Answer:")
            return generated_text[answer_start:].strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Error generating response."