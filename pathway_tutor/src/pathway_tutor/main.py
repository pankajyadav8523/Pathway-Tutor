#!/usr/bin/env python3

from datetime import datetime
from .crew import PathwayTutor
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

def display_welcome():
    print("\n" + "🌟" * 40)
    print("Welcome to Pathway Tutor - Your AI Learning Companion!")
    print("🌟" * 40)
    print("\nI'll help you understand concepts through guided learning.")
    print("Type 'menu' anytime to see options or 'exit' to quit.\n")


def show_menu():
    print("\nOptions:")
    print("1. Ask a new question")
    print("2. Clarify previous answer")
    print("3. Go deeper into current topic")
    print("4. Change subject")
    print("5. Exit")


def handle_response(result, memory):
    print("\n" + "📘 " + "="*50)
    print(f"🧠 Category: {result.get('category', 'General')}")
    print("📝 Guidance:")
    print(result.get('output', 'No guidance generated'))

    follow_up = input("\n🤔 Would you like to: \n1. Ask follow-up \n2. New question \n3. Exit\nChoice (1-3): ")
    return follow_up


def run():
    tutor = PathwayTutor()  # Initialize once outside the loop
    conversation_history = []

    display_welcome()

    while True:
        try:
            question = input("\n💡 What would you like to learn about? ").strip()
            if question.lower() in ['exit', '5']:
                print("\n👋 Thank you for using Pathway Tutor!")
                break

            if question.lower() == 'menu':
                show_menu()
                continue

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
            elif category == "Process-Guide":
                task = tutor.guide_process()
            elif category == "Doubt-Clearing":
                task = tutor.clear_doubt()
            elif category == "Python-Code":
                task = tutor.provide_python_code()
            elif category == "Python-Debug":
                task = tutor.debug_python_code()
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
            

            # Conversation loop for follow-ups
            while True:
                # Store conversation context
                context = {
                    'question': question,
                    'history': "\n".join(conversation_history[-3:]),  # Last 3 exchanges
                    'model': MODEL_NAME
                }
                
                # Run categorization and guidance
                result = tutor.crew().kickoff(inputs=context)
                conversation_history.append(f"Q: {question}\nA: {result.get('output', '')}")
                
                # Handle user flow
                choice = handle_response(result, conversation_history)

                if choice == '1':
                    question = input("\n🔍 What specifically would you like to follow up on? ")
                elif choice == '2':
                    question = input("\n🎯 What new question would you like to ask? ")
                    break
                elif choice == '3':
                    print("\n👋 Thank you for learning with Pathway Tutor!")
                    return
                else:
                    print("\n⚠️ Please choose a valid option.")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Session interrupted. Type 'exit' to quit or ask another question.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            conversation_history.append(f"System Error: {str(e)}")

if __name__ == "__main__":
    run()