#!/usr/bin/env python3

from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
import os
import litellm
from crewai import Crew, Process

# Load environment variables
load_dotenv()

# Configure LiteLLM for Groq
litellm.drop_params = True
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")  # Map Groq key to OpenAI key name
MODEL_NAME = os.getenv("MODEL")

def run():
    tutor = PathwayTutor()  # Initialize once outside the loop
    
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
            
            # Step 1: Categorize the question
            categorize_task = tutor.categorize_question()
            categorize_task.description = categorize_task.description.replace("{question}", question)
            
            # Create a crew for categorization
            category_crew = Crew(
                agents=[tutor.classifier()],
                tasks=[categorize_task],
                process=Process.sequential,
                verbose=True
            )
            category_result = category_crew.kickoff(inputs=inputs)
            category = str(category_result).strip()
            print(f"Question category: {category}")

            # Step 2: Run appropriate task based on category
            if category == "Definition-Based":
                task = tutor.define_term()
            elif category == "Concept-Explanation":
                task = tutor.explain_concept()
            elif category == "Problem-Solving":
                task = tutor.solve_problem()
            elif category == "Comparison":
                task = tutor.compare_concepts()
            else:
                print(f"Category '{category}' not handled yet or irrelevant.")
                continue

            # Update task description with the question
            task.description = task.description.replace("{question}", question) if "{question}" in task.description else task.description

            # Create a crew for the selected task
            execution_crew = Crew(
                agents=[task.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            result = execution_crew.kickoff(inputs=inputs)
            print(f"Result:\n{result}")

        except EOFError:
            print("\nDetected EOF (Ctrl+D). Exiting...")
            break
        except Exception as e:
            print(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()