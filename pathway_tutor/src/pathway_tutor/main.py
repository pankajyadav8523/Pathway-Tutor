#!/usr/bin/env python3
import warnings
import os
from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up Grok API credentials
XAI_API_KEY = os.getenv("XAI_API_KEY")
MODEL_NAME = os.getenv("MODEL", "grok-2-latest")  # Default to latest Grok model

# Initialize OpenAI client for Grok API
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def query_grok(question):
    """Send a user question to Grok and return the response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Grok: {e}"

def run():
    """Main loop for handling user input and interacting with Crew AI and Grok."""
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

            # Call Grok API
            grok_response = query_grok(question)
            print(f"\n🔹 Grok's Response:\n{grok_response}\n")

            # Call Crew AI system
            PathwayTutor().crew().kickoff(inputs=inputs)

        except EOFError:
            print("\nDetected EOF (Ctrl+D). Exiting...")
            break
        except Exception as e:
            print(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
