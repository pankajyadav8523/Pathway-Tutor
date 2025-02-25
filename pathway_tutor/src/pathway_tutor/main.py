#!/usr/bin/env python3

from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
import os
import litellm

# Load environment variables
load_dotenv()

# Configure LiteLLM for Groq
litellm.drop_params = True
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")  # Map Groq key to OpenAI key name
MODEL_NAME = os.getenv("MODEL")

def run():
    while True:
        try:
            question = input("Enter the question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Exiting...")
                break
            
            inputs = {
                'question': question,
                'current_year': str(datetime.now().year),
                'model': MODEL_NAME
            }
            
            PathwayTutor().crew().kickoff(inputs=inputs)
        except EOFError:
            print("\nDetected EOF (Ctrl+D). Exiting...")
            break
        except Exception as e:
            print(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
