import pandas as pd


def first(series):
    return next(series.iteritems())[1]


def last(series):
    l = None
    for item in series:
        l = item
    return l


def count_distinct_blocks(series, basic_block_types_number):
    colors = {"b", "k", "d", "g", "y"}
    if basic_block_types_number == 4:
        basic_blocks = {"f", "l", "r", "s"}
    elif basic_block_types_number == 3:
        basic_blocks = {"f", "l", "r"}
    elif basic_block_types_number == 1:
        basic_blocks = set()

    counts_series = pd.Series(None for _ in range(len(series)))
    for i in series.index:
        counts_series.loc[i] = 0.
        for color in colors:
            if color in series.loc[i]:
                counts_series.loc[i] += 1
                break
        for basic_block in basic_blocks:
            if basic_block in series.loc[i]:
                counts_series.loc[i] += 1
                break
        counts_series.loc[i] += len(set(series.loc[i])
                                    - {"{", "}", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "!", "=", ">", "<"}
                                    - colors - basic_blocks)
    return counts_series


def count_edits(series):
    count = 0
    for item in series:
        if item == "edit":
            count += 1
    return count


def count_submits(series):
    count = 0
    for item in series:
        if item == "execution":
            count += 1
    return count


def count_true(series):
    count = 0
    for item in series:
        if item is True:
            count += 1
    return count


# Counts deletions (1 per shortened line)
def count_deletions(series, consider_multideletions):
    dels = 0
    last = ""
    for item in series:
        if not isinstance(item, str):
            item = ""
        if len(item) < len(last):
            if consider_multideletions:
                pass  # TODO: dels += count_letters("flrsRIW/xdgybk", last) - count_letters("flrsRIW/xdgybk", item)
            else:
                dels += 1
        last = item
    return dels


# Merges snapshots and ts together (left outer join) and returns the result.
def load_extended_snapshots(snapshots_path, task_sessions_path, tasks_path, task_sessions_cols, tasks_cols):
    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=task_sessions_cols)
    tasks = pd.read_csv(tasks_path, usecols=tasks_cols)

    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)
    tasks.rename(index=str, columns={"id": "task"}, inplace=True)

    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")
    snapshots_with_tasks = pd.merge(snapshots_with_tasks, tasks, how="left", on="task")
    return snapshots_with_tasks


def load_task_names_levels(tasks_path):
    tasks = pd.read_csv(tasks_path, usecols=["id", "name", "level"])
    task_names_levels = {task[1].id: {"name": task[1].loc["name"], "level": task[1].level} for task in tasks.iterrows()}
    return task_names_levels


def replace_red_by_d(tasks_path, output_path):
    data = pd.read_csv(tasks_path)
    for i in data.index:
        data["solution"].loc[i] = data["solution"].loc[i].replace("r{", "d{")
    data.to_csv(output_path, index=False)


# Computes flattened lower triangle table (without main diagonal) from a square table
def flattened_triangle_table(table):
    reduced_table = []
    for i in range(len(table)):
        for j in range(i):
            reduced_table.append(table[i][j])
    return reduced_table


"""
replace_red_by_d(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 output_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks4.csv")

load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                        task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                        task_sessions_cols=None)
load_task_names_levels(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")
"""
