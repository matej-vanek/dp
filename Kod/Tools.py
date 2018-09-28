import numpy as np
import pandas as pd
import re


# Counts deletions
# mode: all -> difference of lengths of strings
#       line -> one per each shortened line
#       bit -> one if shortened wherever
def count_deletions(series, mode):
    dels = 0
    last = ""
    for item in series:
        if not isinstance(item, str):
            item = ""
        item = re.sub("[{}0123456789<>=!]", "", item)
        last = re.sub("[{}0123456789<>=!]", "", last)
        if len(item) < len(last):
            if mode == "all":
                dels += len(last) - len(item)
            elif mode == "line":
                dels += 1
            elif mode == "bit":
                dels = 1
        last = item
    return dels


# Counts distinct blocks used in miniRobocode.
# basic_block_types_number determines how many of "f", "l", "r" and "s" blocks collapse into the only one.
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


def entropy(occurence_dict):
    if len(occurence_dict[0]) == 1:
        print(occurence_dict[0].values())
        return 0
    occurence_list = occurence_dict[0].values()
    frequency_list = [i/sum(occurence_list) for i in occurence_list]
    print("{} {} {}".format(occurence_list, frequency_list, 1/np.log2(len(frequency_list)) * sum(map(lambda x: - x * np.log2(x), frequency_list))))
    return 1/np.log2(len(frequency_list)) * sum(map(lambda x: - x * np.log2(x), frequency_list))


# Computes flattened lower triangle table (without main diagonal) from a square table
def flattened_triangle_table(table):
    reduced_table = []
    for i in range(len(table)):
        for j in range(i):
            reduced_table.append(table[i][j])
    return reduced_table


# Counts length of the last item of series
def len_of_last(series):
    return len(re.sub("[{}0123456789<>=!]", "", series.iloc[-1]))


# Merges snapshots and ts together (left outer join) and returns the result.
def load_extended_snapshots(snapshots_path, task_sessions_path, tasks_path, task_sessions_cols, tasks_cols):
    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=task_sessions_cols)
    tasks = pd.read_csv(tasks_path, usecols=tasks_cols)

    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)
    tasks.rename(index=str, columns={"id": "task"}, inplace=True)

    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")
    if tasks_cols:
        snapshots_with_tasks = pd.merge(snapshots_with_tasks, tasks, how="left", on="task")
    return snapshots_with_tasks


def load_task_names_levels(tasks_path):
    tasks = pd.read_csv(tasks_path, usecols=["id", "name", "level"])
    task_names_levels = {task[1].id: {"name": task[1].loc["name"], "level": task[1].level} for task in tasks.iterrows()}
    return task_names_levels


# Computes median of lengths of a string series
def median_of_lens(series):
    return np.median(list(map(lambda x: len(re.sub(pattern="[{}0123456789<>=!]", repl="", string=x)), series)))


# Replaces ambiguous "r" for "right" and "red" by "d" for "red", keeps "r" for "right"
# and saves the dataset
def replace_red_by_d(tasks_path, output_path):
    data = pd.read_csv(tasks_path)
    for i in data.index:
        data["solution"].loc[i] = data["solution"].loc[i].replace("r{", "d{")
    data.to_csv(output_path, index=False)


# Creates dict of solutions and number of their occurences
def solutions_dict(series):
    solutions = {}
    for item in series:
        if item not in solutions:
            solutions[item] = 0
        solutions[item] += 1
    print(solutions)
    return [solutions]


"""
replace_red_by_d(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 output_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks4.csv")

load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                        task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                        task_sessions_cols=None)
load_task_names_levels(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")
"""
