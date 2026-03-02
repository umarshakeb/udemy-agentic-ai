# 🤖 Udemy Agentic AI Research Crew

An autonomous multi-agent system designed to perform deep-market research on Advancements in AI and LLMs in medical field. Built with **CrewAI** and powered by local LLMs via **Ollama**.

## 🚀 Key Features
- **Multi-Agent Orchestration:** Utilizes a Senior Data Researcher and a Reporting Analyst to decompose complex research tasks.
- **Local LLM Integration:** Runs **Qwen 2.5 (3B)** locally via Ollama to ensure data privacy and zero API costs.
- **AI Observability:** Integrated with **OpenInference** and **LiteLLM** for real-time tracing of agent "thought" trajectories.
## Future Work
- **GPU Optimized:** Configuration for NVIDIA CUDA acceleration to handle high-token throughput on consumer hardware.

## 🛠️ Tech Stack
- **Framework:** CrewAI
- **Inference:** Ollama (Local)
- **Monitoring:** OpenInference / OpenTelemetry
- **Environment:** Python (Managed by `uv`)

## 📋 How to Run
1. Ensure Ollama is running: `ollama run qwen2.5:3b`
2. Install dependencies: `uv sync`
3. Run the crew: `crewai run`


## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/udemyagentic/config/agents.yaml` to define your agents
- Modify `src/udemyagentic/config/tasks.yaml` to define your tasks
- Modify `src/udemyagentic/crew.py` to add your own logic, tools and specific args
- Modify `src/udemyagentic/main.py` to add custom inputs for your agents and tasks

