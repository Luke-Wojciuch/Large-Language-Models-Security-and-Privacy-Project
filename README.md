# Large-Language-Models-Security-and-Privacy-Project

To run this project, ensure Ollama is installed with Qwen3.5:9b available on port "http://localhost:11434/v1" with the default API key "Ollama".

Running full_experiment.py will run the tool_knowledge benchmark with all user tasks and all injection tasks across the banking and slack suites. This is not recommended: it took 16 hours on a GTX 1080 TI. To change the number of tasks or injection tasks for any specific experimental condition, change the NUM_TASKS and NUM_INJECTION_TASKS variables from None to an integer value, ideally below 4.