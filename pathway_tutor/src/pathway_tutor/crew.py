from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.llms import OpenAI
from .memory import PathwayMemory

@CrewBase
class PathwayTutor(Crew):
    """PathwayTutor AI Crew"""
    def __init__(self):
        self.memory = PathwayMemory()

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def _create_agent(self, config_name):
        return Agent(
            config=self.agents_config[config_name],
            verbose=True,
            memory=self.memory,  # Inject memory
            llm=OpenAI(temperature=0),  # Base LLM for memory ops
            allow_delegation=False,
            max_iter=3
        )
    
    @agent
    def classifier(self) -> Agent:
        return self._create_agent('classifier')

    @agent
    def definition_based(self) -> Agent:
        return self._create_agent('definition_based')

    @agent
    def concept_explanation(self) -> Agent:
        return self._create_agent('concept_explanation')

    @agent
    def problem_solving(self) -> Agent:
        return self._create_agent('problem_solving')

    @agent
    def comparison(self) -> Agent:
        return self._create_agent('comparison')

    @agent
    def process_guide(self) -> Agent:
        return self._create_agent('process_guide')

    @agent
    def doubt_clearing(self) -> Agent:
        return self._create_agent('doubt_clearing')

    @agent
    def python_code(self) -> Agent:
        return self._create_agent('python_code')

    @agent
    def python_debug(self) -> Agent:
        return self._create_agent('python_debug')

    @task
    def categorize_question(self) -> Task:
        """Task to classify the question into a category"""
        return Task(
            config=self.tasks_config['categorization'],
            agent=self.classifier()
        )

    @task
    def define_term(self) -> Task:
        """Task to provide a definition and basic explanation of a term"""
        return Task(
            config=self.tasks_config['definition_based_tasks']['task_1'],
            agent=self.definition_based()
        )

    @task
    def explain_concept(self) -> Task:
        """Task to provide a detailed explanation of a concept"""
        return Task(
            config=self.tasks_config['concept_explanation_tasks']['task_1'],
            agent=self.concept_explanation()
        )

    @task
    def solve_problem(self) -> Task:
        """Task to solve a problem step-by-step"""
        return Task(
            config=self.tasks_config['problem_solving_tasks']['task_1'],
            agent=self.problem_solving()
        )

    @task
    def compare_concepts(self) -> Task:
        """Task to compare and contrast concepts"""
        return Task(
            config=self.tasks_config['comparison_tasks']['task_1'],
            agent=self.comparison()
        )

    @task
    def guide_process(self) -> Task:
        """Task to guide through a process"""
        return Task(
        config=self.tasks_config['process_guide_tasks']['task_1'],
        agent=self.process_guide()
        )

    @task
    def clear_doubt(self) -> Task:
        """Task to clear a doubt"""
        return Task(
        config=self.tasks_config['doubt_clearing_tasks']['task_1'],
        agent=self.doubt_clearing()
        )

    @task
    def provide_python_code(self) -> Task:
        """Task to provide Python code"""
        return Task(
        config=self.tasks_config['python_code_tasks']['task_1'],
        agent=self.python_code()
        )

    @task
    def debug_python_code(self) -> Task:
        """Task to debug Python code"""
        return Task(
        config=self.tasks_config['python_debug_tasks']['task_1'],
        agent=self.python_debug()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PathwayTutor crew"""
        return Crew(
            agents=[
                self.classifier(),
                self.definition_based(),
                self.concept_explanation(),
                self.problem_solving(),
                self.comparison(),
                self.process_guide(),
                self.doubt_clearing(),
                self.python_code(),
                self.python_debug()
            ],
            tasks=[
                self.categorize_question(),
                self.define_term(),
                self.explain_concept(),
                self.solve_problem(),
                self.compare_concepts(),
                self.guide_process(),
                self.clear_doubt(),
                self.provide_python_code(),
                self.debug_python_code()

            ],
            memory=self.memory,
            process=Process.sequential,
            verbose=2,
            full_output=True
        )