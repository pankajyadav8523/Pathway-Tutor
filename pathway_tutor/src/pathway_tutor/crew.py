from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class PathwayTutor():
    """PathwayTutor AI Crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def classifier(self) -> Agent:
        """Agent responsible for categorizing the student's question"""
        return Agent(
            config=self.agents_config['classifier'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    @agent
    def definition_based(self) -> Agent:
        """Agent responsible for providing definitions and basic explanations"""
        return Agent(
            config=self.agents_config['definition_based'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    @agent
    def concept_explanation(self) -> Agent:
        """Agent responsible for providing detailed explanations of concepts"""
        return Agent(
            config=self.agents_config['concept_explanation'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    @agent
    def problem_solving(self) -> Agent:
        """Agent responsible for solving problems step-by-step"""
        return Agent(
            config=self.agents_config['problem_solving'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    @agent
    def comparison(self) -> Agent:
        """Agent responsible for comparing and contrasting concepts"""
        return Agent(
            config=self.agents_config['comparison'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    @agent
    def process_guide(self) -> Agent:
        """Agent responsible for guiding through processes"""
        return Agent(
            config=self.agents_config['process_guide'],
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    @agent
    def doubt_clearing(self) -> Agent:
        """Agent responsible for clearing doubts"""
        return Agent(
        config=self.agents_config['doubt_clearing'],
        verbose=True,
        allow_delegation=False,
        max_iter=3
        )

    @agent
    def python_code(self) -> Agent:
        """Agent responsible for providing Python code solutions"""
        return Agent(
        config=self.agents_config['python_code'],
        verbose=True,
        allow_delegation=False,
        max_iter=3
        )

    @agent
    def python_debug(self) -> Agent:
        """Agent responsible for debugging Python code"""
        return Agent(
        config=self.agents_config['python_debug'],
        verbose=True,
        allow_delegation=False,
        max_iter=3
        )

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
            process=Process.sequential,
            verbose=True,
        )