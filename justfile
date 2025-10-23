one PART = "":
    # (cd ./src/task && ../../.venv/bin/python -c "import task_one; task_one.task_one()")
    .venv/bin/python -m src.task.task_one "{{PART}}"

two PART = "":
    # (cd ./src/task && ../../.venv/bin/python -c "import task_two; task_two.task_two()")
    .venv/bin/python -m src.task.task_two "{{PART}}"

three PART = "":
    # (cd ./src/task && ../../.venv/bin/python -c "import task_three; task_three.task_three()")
    .venv/bin/python -m src.task.task_three "{{PART}}"

four PART = "":
    # (cd ./src/task && ../../.venv/bin/python -c "import task_four; task_four.task_four()")
    .venv/bin/python -m src.task.task_four "{{PART}}"