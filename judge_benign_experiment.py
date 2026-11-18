
from pathlib import Path
from openai import OpenAI
from pprint import pprint
from functools import wraps

from agentdojo.agent_pipeline.agent_pipeline import (
    AgentPipeline,
    PipelineConfig,
)
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.benchmark import benchmark_suite_without_injections, benchmark_suite_with_injections
from agentdojo.task_suite.load_suites import get_suite, get_suites
from experiment_progress import ProgressLogger
from experiment_tasks import select_tasks
from agentdojo.attacks.attack_registry import load_attack

from permission_judge import (
    PermissionJudge,
    JudgedToolsExecutor,
)

def judge_benign_experiment(
    tasks=None, *, model, version, logdir, suites, action_prompt
):
    print("\nRUNNING JUDGE BENIGN EXPERIMENT")

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Connect to the local model through Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Use AgentDojo's existing LLM implementation
    llm = OpenAILLM(client, model)

    # Build an unprotected baseline agent
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=model,
            defense=None,
            system_message_name=None,
            system_message=action_prompt,
        )
    )
    # Use a separate API client for the permission judge.
    judge_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # The permission judge
    judge = PermissionJudge(
        client=judge_client,
        model=model,
    )

    # find the tool-execution loop
    tools_loop = next(
        element for element in pipeline.elements
        if isinstance(element, ToolsExecutionLoop)
    )

    # Replace ONLY the executor, preserving the agent LLM
    executor_index = next(
        i for i, element in enumerate(tools_loop.elements)
        if isinstance(element, ToolsExecutor)
    )
    tools_loop.elements[executor_index] = JudgedToolsExecutor(
        judge=judge
    )

    # Distinct identity for experimental results
    pipeline.name = f"local-{model.replace(':', '-')}-permission-judge"

    all_results = {}

    # Execute each suite sequentially
    with ProgressLogger(str(logdir)) as progress:

        for suite_name in suites:

            suite = get_suite(version, suite_name)

            selected_tasks = select_tasks(suite.user_tasks, tasks)

            progress.start_suite(suite, selected_tasks)
            results = benchmark_suite_without_injections(
                agent_pipeline=pipeline,
                suite=suite,
                logdir=logdir,
                force_rerun=True,
                user_tasks=selected_tasks,
                benchmark_version=version,
            )

            all_results[suite_name] = results

    # Summarize utility
    print("\n========== RESULTS ==========")

    total_passed = 0
    total_attacker_successful = 0
    total_tasks = 0

    print(results)

    for suite_name, results in all_results.items():

        utility = results["utility_results"]
        security = results["security_results"]

        passed = sum(utility.values())
        total = len(utility)

        attacker_successful = sum(security.values())
        if total != len(security):
            print(f"Mismatch in Utility and Security for suite {suite_name}.")
        

        total_passed += passed
        total_attacker_successful += attacker_successful
        total_tasks += total

        print(
            f"{suite_name} Utility: "
            f"{passed}/{total} "
            f"({passed / total:.1%})"
        )

    print("\nOverall utility:")
    print(f"{total_passed}/{total_tasks}")
    print(f"{total_passed / total_tasks:.1%}")


if __name__ == "__main__":
    from full_experiment import EXPERIMENT_OPTIONS, LOG_DIRS

    judge_benign_experiment(
        logdir=LOG_DIRS["judge_benign"], **EXPERIMENT_OPTIONS
    )
