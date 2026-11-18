# Large Language Models Security and Privacy Project

This project uses AgentDojo to compare a tool-using agent with and without an LLM permission judge. The checked-in benchmark results and historical console output are in [runs](runs).

## Setup

Use Python 3.10 or newer and install the benchmark dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
```

The defaults expect Ollama to serve `qwen3.5:9b` at `http://localhost:11434/v1`. You can also configure an OpenAI-compatible endpoint that supports tool calling for the agent and JSON response mode for the judge. Set `EXPERIMENT_API_KEY` in the environment for a hosted endpoint. Use `judge_api_key_env` when the judge needs a separate credential.

## Configure and run from one notebook cell

Once the repository is importable and the dependencies and model endpoint are available, this cell runs a small benign comparison. The output path below is writable on Kaggle. For local use, change it to `Path("runs/benign")`.

```python
from pathlib import Path

from experiment import ExperimentConfig, print_summary, run_experiment

config = ExperimentConfig(
    suites=("banking", "slack"),
    conditions=("baseline", "judge"),
    task_limit=2,                      # Per suite. Use None for all tasks.
    benchmark_version="v1.2",
    output_dir=Path("/kaggle/working/agentdojo-runs"),
    model="qwen3.5:9b",
    base_url="http://localhost:11434/v1",
    api_key_env="EXPERIMENT_API_KEY",
    judge_model=None,                  # Inherit the agent model.
    judge_base_url=None,               # Inherit the agent endpoint.
    judge_api_key_env=None,            # Inherit the agent credential setting.
    timeout=120,                       # Seconds per API request.
    max_retries=2,
    force_rerun=True,
)

results = run_experiment(config)
print_summary(results)
```

On Kaggle, `localhost` refers to the notebook runtime. Configure an endpoint reachable from that runtime, and load hosted credentials from Kaggle Secrets into the named environment variables before running the cell. Set `system_prompt` on the config to customize the agent prompt. The returned results are keyed by condition and suite. AgentDojo writes task logs below the configured output directory. If you enable result reuse, use the same output directory only for equivalent model and prompt settings.

## Command-line entry points

- `python experiment.py --help` lists the options for the configurable benign runner.
- `python experiment.py --suites banking --conditions baseline judge --task-limit 2` runs a quick comparison. Use `--all-tasks` for the entire selected suite.
- `full_experiment.py` is the historical four-condition runner. Its original full run took about 16 hours on a GTX 1080 Ti.
- `demo.py` is a separate LangChain demonstration. `qwen_response_test.py` is a direct endpoint smoke check.
