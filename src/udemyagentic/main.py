#!/usr/bin/env python
import sys
import warnings
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor
from datetime import datetime
from termcolor import colored
import asyncio
from rich.console import Console
from udemyagentic.crew import Udemyagentic
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

 
langfuse = get_client()
CrewAIInstrumentor().instrument(skip_deck_check=True)
LiteLLMInstrumentor().instrument()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
 
def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Advancements in AI and LLMs in medical field',
        'current_year': str(datetime.now().year)
    }

    with langfuse.start_as_current_span(name="crew_execution_span_trace"):
        try:
            Udemyagentic().crew().kickoff(inputs=inputs)
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")
    langfuse.flush()  # Ensure all telemetry data is sent to the server

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Udemyagentic().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Udemyagentic().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        Udemyagentic().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = Udemyagentic().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
