from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import (Any, List)
from pydantic import BaseModel, Field
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

import litellm
litellm.telemetry = False
litellm.success_callback = []
litellm.failure_callback = []

llm=LLM(model="ollama/qwen-gpu",
    base_url="http://localhost:11434",
    api_key="ollama",
    extra_headers={
        "num_gpu": "35",  # 35 is usually the total layers for a 3B model; forces all to GPU
        "main_gpu": "0"
    })


class Section(BaseModel):
    title: str = Field(..., description="The title of the section")
    overview: str = Field(description="Overview of the title")
    keyDevelopments: list[str] = Field(description="Key informations from the title")
    impact: str = Field(description="Impact on users because of the title")

class ResearchReport(BaseModel):
    title:str = Field(description="Title of the report")
    sections:list[Section] = Field(description="List of sections together forming a report")
    conclusion:str = Field(description="Conclusion of the report")

class ResearchPoints(BaseModel):
   sections: list[str] = Field(description="List of bullet points together forming a report")


@CrewBase
class Udemyagentic():
    """Udemyagentic crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            llm = llm
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            llm = llm
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            # output_pydantic=ResearchPoints
            output_json=ResearchPoints
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            # output_file='report.md'
            output_file='report.json'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Udemyagentic crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
