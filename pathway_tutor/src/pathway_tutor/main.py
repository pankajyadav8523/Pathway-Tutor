#!/usr/bin/env python3
import warnings
from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# model = os.getenv("MODEL")

# litellm.completion(api_key=api_key, model=model)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew to categorize a student's question.
    """
    while True:
        try:
            question = input("Enter the question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Exiting...")
                break
            
            inputs = {
                'question': question,
                'current_year': str(datetime.now().year)
            }
            
            PathwayTutor().crew().kickoff(inputs=inputs)
        except EOFError:
            print("\nDetected EOF (Ctrl+D). Exiting...")
            break
        except Exception as e:
            print(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
