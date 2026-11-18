from agentdojo.logging import OutputLogger


class ProgressLogger(OutputLogger):
    def start_suite(self, suite, user_tasks, injection_tasks=None, attack=None):
        user_count = len(suite.user_tasks if user_tasks is None else user_tasks)
        self.total = user_count
        if attack is not None and not attack.is_dos_attack:
            injection_count = len(
                suite.injection_tasks if injection_tasks is None else injection_tasks
            )
            self.total = (user_count + 1) * injection_count
        self.current = 0
        self.previous_case = None

    def log(self, messages, **context):
        user_task = context["user_task_id"]
        injection_task = context["injection_task_id"]
        case = (user_task, injection_task)
        if case != self.previous_case:
            self.previous_case = case
            self.current += 1
            if injection_task is None and user_task.startswith("injection_task_"):
                label = f"Injection baseline {user_task}"
            else:
                label = f"User task {user_task}"
                if injection_task is not None:
                    label += f" | Injection task {injection_task}"
            print(
                f"[{context['suite_name']} {self.current}/{self.total}] {label}",
                flush=True,
            )
        super().log(messages, **context)
