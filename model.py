import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """
    Load the model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the pre-trained model directory.

    Returns:
        ModelWrapper: A wrapper instance for handling model interactions.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode for inference
        return ModelWrapper(model, tokenizer)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, text):
        """
        Generate a response from the input text.

        Args:
            text (str): The input text for the model to process.

        Returns:
            str: The generated response as a string.
        """
        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Failed to generate response: {e}"
