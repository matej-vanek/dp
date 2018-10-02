from functools import partial
import numpy as np
import pandas as pd
import re
from scipy.cluster.hierarchy import fcluster, linkage

from AST import *
from MiniRoboCodeInterpreter import run_task


def add_new_run_and_square_sequence(snapshots_path, task_sessions_path, tasks_path, output_snapshots_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "task"],
                                   tasks_cols=[])
    data["new_correct"] = None
    data["square_sequence"] = None
    for i in data.index:
        if data.loc[i]["granularity"] == "execution":
            #print(data.loc[i]["program"])
            correct, square_sequence = run_task(tasks_path=tasks_path,
                                                task_id=data.loc[i]["task"],
                                                program=data.loc[i]["program"],
                                                verbose=False)
            data.set_value(i, "new_correct", str(correct))
            data.set_value(i, "square_sequence", square_sequence)
    data = data.drop(["task"], axis=1)
    data.to_csv(output_snapshots_path, index=False)


def contains_true(series):
    return series


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


def count_program_clusters(programs):
    clusters_count = pd.Series(index=programs.index)
    for task in programs.index:
        if len(programs.loc[task][0].keys()) > 1:
            condensed_dist_matrix = []
            program_list = list(programs.loc[task][0].keys())
            print(program_list)
            program_list = np.array(list(map(partial(build_ast), program_list)))
            #print(program_list)
            for i in range(len(program_list)):
                for j in range(len(program_list)):
                    if i < j:
                        condensed_dist_matrix.append(ast_ted(program_list[i], program_list[j]))
            print(condensed_dist_matrix)
            condensed_dist_matrix = np.ndarray.flatten(np.array(condensed_dist_matrix))
        else:
            condensed_dist_matrix = [0]
        hier_clust = linkage(condensed_dist_matrix)
        print(hier_clust)

        cluster_assign = fcluster(hier_clust, 5, criterion="distance")
        print(cluster_assign)
        print("Number of found clusters: ", len(set(cluster_assign)))
        clusters_count.loc[task] = len(set(cluster_assign))
    return clusters_count


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


# Creates dict of solutions and number of their occurences
def dict_of_counts(series):
    solutions = {}
    for item in series:
        if item not in solutions:
            solutions[item] = 0
        solutions[item] += 1
    #print(solutions)
    return [solutions]


def entropy(occurence_dict):
    if len(occurence_dict[0]) == 1:
        #print(occurence_dict[0].values())
        return 0
    occurence_list = occurence_dict[0].values()
    frequency_list = [i/sum(occurence_list) for i in occurence_list]
    #print("{} {} {}".format(occurence_list, frequency_list, 1/np.log2(len(frequency_list)) * sum(map(lambda x: - x * np.log2(x), frequency_list))))
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


# Merges snapshots, task_sessions and tasks together (left outer join) and returns the result.
def load_extended_snapshots(snapshots_path, task_sessions_path, tasks_path, task_sessions_cols, tasks_cols):
    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=task_sessions_cols)
    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)
    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")

    if tasks_cols:
        tasks = pd.read_csv(tasks_path, usecols=tasks_cols)
        tasks.rename(index=str, columns={"id": "task"}, inplace=True)
        snapshots_with_tasks = pd.merge(snapshots_with_tasks, tasks, how="left", on="task")
    return snapshots_with_tasks


# Merges snapshots and ts together (left outer join) and returns the result.
def load_extended_task_sessions(task_sessions_path, snapshots_path, snapshots_cols):
    task_sessions = pd.read_csv(task_sessions_path)
    snapshots = pd.read_csv(snapshots_path, usecols=snapshots_cols)
    snapshots.rename(index=str, columns={"id": "snapshot", "task_session": "id"}, inplace=True)
    task_sessions_with_snapshots = pd.merge(task_sessions, snapshots, how="left", on="task_session")
    return task_sessions_with_snapshots


def load_task_names_levels(tasks_path):
    tasks = pd.read_csv(tasks_path, usecols=["id", "name", "level"])
    task_names_levels = {task[1].id: {"name": task[1].loc["name"], "level": task[1].level} for task in tasks.iterrows()}
    return task_names_levels


# Computes median of lengths of a string series
def median_of_lens(series):
    return np.median(list(map(lambda x: len(re.sub(pattern="[{}0123456789<>=!]", repl="", string=str(x))), series)))


# Replaces ambiguous "r" for "right" and "red" by "d" for "red", keeps "r" for "right"
# and saves the dataset
def replace_red_by_d(tasks_path, output_path, column_name):
    data = pd.read_csv(tasks_path)
    for i in data.index:
        if isinstance(data[column_name].loc[i], str):
            #print(data[column_name].loc[i])
            data[column_name].loc[i] = data[column_name].loc[i].replace("r{", "d{")
    data.to_csv(output_path, index=False)


# Determines whether the sample solution is the most used correct solution
def sample_solution_most_frequent(solutions, programs):
    output = pd.Series(index=solutions.index)
    for i in solutions.index:
        #print(max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]))
        if solutions.loc[i] == max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]).replace("r{", "d{"):
            output.loc[i] = True
        else:
            #print(i, solutions.loc[i], max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]))
            output.loc[i] = False
    return output


"""
replace_red_by_d(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 output_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots2.csv",
                 column_name="program")
"""
"""
load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                        task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                        task_sessions_cols=None)
load_task_names_levels(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")

add_new_run_and_square_sequence(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                                task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                                tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                                output_snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots_2.csv")
"""