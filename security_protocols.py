# ai_engine/models.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

class AIModel:
    def __init__(self, model_name='t5-small'):
        """
        Initializes the AI model.

        Args:
            model_name (str): Name of the pre-trained model to load.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate_command(self, input_text):
        """
        Generates a command based on input text.

        Args:
            input_text (str): The input text.

        Returns:
            str: Generated command.
        """
        inputs = self.tokenizer.encode(f"Translate to command: {input_text}", return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
        command = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return command
