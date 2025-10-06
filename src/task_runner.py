from task.task_one import task_one
from task.task_two import task_two
from task.task_three import task_three
from task.task_four import task_four

def run_task(task: int = 0, part: str = None):
    if task == 1:
        task_one(part)
    elif task == 2:
        task_two(part)
    elif task == 3:
        task_three(part)
    elif task == 4:
        task_four(part)
    elif task == 0:
        task_one(part)
        task_two(part)
        task_three(part)
        task_four(part)