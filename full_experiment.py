from judge_malicious_experiment import judge_malicious_experiment
from judge_benign_experiment import judge_benign_experiment
from malicious_experiment import standard_malicious_experiment
from benign_experiment import standard_benign_experiment



if __name__ == "__main__":

    standard_benign_experiment(tasks=None)

    judge_benign_experiment(tasks=None)

    standard_malicious_experiment(tasks=None, injection_tasks=None)

    judge_malicious_experiment(tasks=None, injection_tasks=None)