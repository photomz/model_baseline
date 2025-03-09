from .open_ai import OpenAIAdapter
import os
from groq import Groq
import openai


# Groq should be superset of OpenAI
class GroqAdapter(OpenAIAdapter):
    def init_client(self):
        """
        Initialize the Groq client
        """
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")

        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        # client = Groq()
        return client


if __name__ == "__main__":
    adapter = GroqAdapter(config="qwen-qwq-32b")
    attempt = adapter.make_prediction(prompt="What is the capital of France?")
    print(attempt)
