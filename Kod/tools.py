#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from ast import literal_eval
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from robomission_ast import *
from minirobocode_interpreter import run_task


# Runs submits again and tests their correctness.
# Computes sequence of visited squares during the run.
def synchronous_interpreter_correctness_and_square_sequence(snapshots_path=None, task_sessions_path=None, tasks_path=None, output_snapshots_path=None, only_executions=True, only_edits=False, dataframe=None, save=True):
    print("Interpreter")
    if dataframe is not None:
        data = dataframe
    else:
        data = load_extended_snapshots(snapshots_path=snapshots_path,
                                       task_sessions_path=task_sessions_path,
                                       tasks_path=tasks_path,
                                       task_sessions_cols=["id", "task"],
                                       tasks_cols=[])
        data["new_correct"] = pd.Series(None, index=data.index)
        data["square_sequence"] = pd.Series(None, index=data.index)
    print(data.shape[0])
    for i in data.index:
        if only_executions:
            if data.loc[i].granularity != "execution":
                continue
        if only_edits:
            if data.loc[i].granularity != "edit":
                continue
        #print(data.loc[i]["program"])
        print(i)
        correct, square_sequence = run_task(tasks_path=tasks_path,
                                            task_id=data.loc[i].task,
                                            program=data.loc[i].program,
                                            verbose=False)

        #print(square_sequence)
        data.new_correct.loc[i] = str(correct)
        data.square_sequence.loc[i] = str(square_sequence)
        #print(data.loc[i].square_sequence)
    if save:
        data = data.drop(["task"], axis=1)
        data.to_csv(output_snapshots_path, index=False)
    return data


# Counts bag of blocks characteristics of tasks.
def bag_of_blocks(series):
    blocks = {"f": 0, "l": 1, "r": 2, "s": 3, "R": 4, "W": 5, "I": 6, "/": 7, "x": 8,
              "y": 9, "b": 9, "k": 9, "d": 9, "g": 9}  # color condition does not have its own symbol
    output = []
    for i in series.index:
        program = series.loc[i]
        block_counts = [0 for _ in range(10)]
        for char in program:
            if char in blocks:
                block_counts[blocks[char]] += 1


        output.append(block_counts)
    output = pd.Series(output, index=series.index)
    return output


# Counts bag of task game-world entities.
def bag_of_entities(task_setting_series):
    # b, k, y, g, d(r) -> all, y-g-d (colorful)
    # D, A, M, W, X, Y, Z, (S) -> wormholes, diamonds, asteroids, meteoroids
    bag = []
    for i in task_setting_series.index:
        task = literal_eval(task_setting_series.loc[i])
        task = re.sub("r", "d", task["fields"])

        entities = [0 for _ in range(6)]  # [size, colorful, diamonds, wormholes, asteroids, meteoroids]
        entities[0] = len(re.findall("[bkygd]", task))
        entities[1] = len(re.findall("[ygd]", task))
        entities[2] = len(re.findall("D", task))
        entities[3] = len(re.findall("[WXYZ]", task))
        entities[4] = len(re.findall("A", task))
        entities[5] = len(re.findall("M", task))
        bag.append(entities)
    bag = pd.Series(bag, index=task_setting_series.index)
    return bag


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


# Builds ASTs from programs, computes their TED matrix, hierarchically clusters them,
# prunes where cophenetic dist is > 5, returns number of clusters
def count_program_clusters2(programs):
    clusters_count = pd.Series(index=programs.index)
    program_info = {}
    cluster_info = {}
    for task in programs.index:
        program_list = list(programs.loc[task][0].keys())
        if len(programs.loc[task][0].keys()) > 1:
            condensed_dist_matrix = []
            print(len(program_list))
            program_ast_list = np.array(list(map(partial(build_ast), program_list)))
            #print(program_ast_list)
            for i in range(len(program_ast_list)):
                print(i)
                for j in range(len(program_ast_list)):
                    if i < j:
                        condensed_dist_matrix.append(ast_ted(program_ast_list[i], program_ast_list[j]))
            #print(condensed_dist_matrix)
            condensed_dist_matrix = np.ndarray.flatten(np.array(condensed_dist_matrix))
        else:
            condensed_dist_matrix = [0]

        hier_clust = linkage(condensed_dist_matrix)
        #print(hier_clust)
        #print(programs.loc[task])
        cluster_assign = fcluster(hier_clust, 5, criterion="distance")
        #print(program_list)
        #cluster_info[task] = {cluster:
        #                          {'programs': [program for program in program_info[task] if program["cluster"] == cluster],
        #                           'representant': max(program_info[task][0], key=program_info[task][0].get)}
        #                           for cluster in set(cluster_assign)}

        print(set(cluster_assign))
        for cluster in set(cluster_assign):
            print(cluster)
            print(program_info[task])
            programs = [program_info[task][program] for program in program_info[task] if
                                                program_info[task][program]["cluster"] == cluster]
            representant = max(program_info[task][0], key=program_info[task][0].get)

            cluster_info[task] = {cluster: {"programs": programs, "representant": representant}}

        #cluster_info[task] = {cluster:
        #                          {'programs': [program_info[task][program] for program in program_info[task] if
        #                                        program_info[task][program]["cluster"] == cluster],
        #                           'representant': max(program_info[task][0], key=program_info[task][0].get)}
        #                      for cluster in set(cluster_assign)}



        #print(cluster_assign)
        #print("Number of found clusters: ", len(set(cluster_assign)))
        clusters_count.loc[task] = len(set(cluster_assign))
        print(program_info[task])
        print(cluster_info)
        print(cluster_info[task])
    return clusters_count, program_info, cluster_info

def count_program_clusters(programs):
    clusters_count = pd.Series(index=programs.index)
    program_info = {}
    cluster_info = {}
    for task in programs.index:
        program_list = list(programs.loc[task][0].keys())
        if len(programs.loc[task][0].keys()) > 1:
            condensed_dist_matrix = []
            #print(len(program_list))
            program_ast_list = np.array(list(map(partial(build_ast), program_list)))
            #print(program_ast_list)
            for i in range(len(program_ast_list)):
                #print(i)
                for j in range(len(program_ast_list)):
                    if i < j:
                        condensed_dist_matrix.append(ast_ted(program_ast_list[i], program_ast_list[j]))
            #print(condensed_dist_matrix)
            condensed_dist_matrix = np.ndarray.flatten(np.array(condensed_dist_matrix))
        else:
            condensed_dist_matrix = [0]
        hier_clust = linkage(condensed_dist_matrix)
        #print(hier_clust)
        #print(programs.loc[task])
        cluster_assign = fcluster(hier_clust, 5, criterion="distance")
        #print(program_list)
        program_info[task] = {program: {'cluster': cluster, 'freq': programs.loc[task][0][program]} for program, cluster in zip(program_list, cluster_assign)}
        #print(program_info[task])
        #print([prog for prog in program_info[task] if program_info[task][prog]["cluster"] == 1])

        cluster_info[task] = {cluster: {'programs': [program for program in program_info[task] if program_info[task][program]["cluster"] == cluster],
                                        'representant': max([prog for prog in program_info[task] if program_info[task][prog]["cluster"] == cluster], key=lambda x: program_info[task][x]["freq"])}
                              for cluster in set(cluster_assign)}
        #print(cluster_assign)
        #print("Number of found clusters: ", len(set(cluster_assign)))
        clusters_count.loc[task] = len(set(cluster_assign))
        #print(cluster_info[task])
    return clusters_count, program_info, cluster_info


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


def count_frequent_wrong_programs_ratio(tasks, abs_threshold, rel_threshold):
    frequency = pd.Series(index=tasks.index.levels[0])
    unique = pd.Series(index=tasks.index.levels[0])
    frequent_programs_series = pd.Series(index=tasks.index.levels[0])
    for task in tasks.index.levels[0]:
        frequent_ratio = 0
        unique_programs = 0
        frequent_programs = [{}]
        for seq in tasks.index:
            if seq[0] == task:
                this_seq = tasks.loc[seq]
                if this_seq.abs_count >= abs_threshold and \
                        this_seq.rel_count >= rel_threshold and \
                        isinstance(this_seq.most_frequent_program, str):
                    frequent_ratio += this_seq.rel_count
                    unique_programs += 1
                    frequent_programs[0][this_seq.most_frequent_program] = this_seq.abs_count
        frequency.loc[task] = frequent_ratio
        unique.loc[task] = unique_programs
        frequent_programs_series.loc[task] = frequent_programs

    return frequency, unique, frequent_programs_series


def count_total_abs_freq(series):
    abs_freq = pd.Series(index=series.index)
    for i in series.index:
        abs_freq.loc[i] = sum([series.loc[i][0][key] for key in series.loc[i][0]])
    return abs_freq


# Counts how many tasks in the distance matrix have lower distance to the source task than threshold.
# Computes for all rows of the distance matrix.
def count_similar_tasks(distance_matrix, threshold):
    output = pd.Series(index=distance_matrix.index)
    for i in distance_matrix.index:
        x = [1 for task in distance_matrix if distance_matrix.loc[i, task] <= threshold]
        output.loc[i] = sum(x)
    return output


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


def count_task_frequency(dataframe):
    #print(dataframe.index)

    task_freq_series = pd.Series(index=dataframe.index)
    for i in dataframe.index.levels[0]:
        task_freq = 0
        for j in dataframe.index:
            if j[0] == i:
                #print(dataframe.loc[j])
                task_freq += dataframe.loc[j].abs_count
        for j in dataframe.index:
            if j[0] == i:
                task_freq_series.loc[j] = task_freq
    return task_freq_series



# counts all block types used in all items of series
def count_used_blocks(series):
    blocks = set()
    for i in series.index:
        for char in series.loc[i]:
            if char in "0123456789{}=!<>":
                continue
            elif char in "dbkyg":
                blocks.add("color")
            else:
                blocks.add(char)
    return len(blocks)


# Creates dict of solutions and number of their occurences
def dict_of_counts(series, del_false=False):
    solutions = {}
    for item in series:
        if item not in solutions:
            solutions[item] = 0
        solutions[item] += 1
    if del_false:
        if False in solutions:
            del solutions[False]
    #print(solutions)
    return [solutions]


def entropy(occurence_dict):
    if len(occurence_dict[0]) == 1:
        #print(occurence_dict[0].values())
        return 0
    occurence_list = occurence_dict[0].values()
    frequency_list = [i/sum(occurence_list) for i in occurence_list]
    return 1/np.log2(len(frequency_list)) * sum([- x * np.log2(x) for x in frequency_list])


# Flattens table and omits None values
def flatten_table_remove_nan(table, triangle=False):
    output = []
    for i in table.index:
        for j in table:
            if table.loc[i, j] is not np.nan and table.loc[i, j] is not None:
                if triangle:
                    if i < j:
                        output.append(table.loc[i, j])
                else:
                    output.append(table.loc[i, j])
    return output


# Computes flattened lower triangle table (without main diagonal) from a square table
def flattened_triangle_table(table):
    reduced_table = []
    for i in range(len(table)):
        for j in range(i):
            reduced_table.append(table[i][j])
    return reduced_table


def get_most_frequent_program(program_series):
    most_freq_program = pd.Series(index=program_series.index)
    for i in program_series.index:
        #print(program_series.loc[i])
        most_freq_program.loc[i] = max(program_series.loc[i][0], key=lambda x: program_series.loc[i][0][x])
    return most_freq_program



def get_relative_counts(abs_counts):
    output = pd.Series(index=abs_counts.index)
    total_sum = 0
    for i in abs_counts.index:
        task_rel_counts = {}
        programs = abs_counts.loc[i][0]
        task_sum = sum(programs.values())
        for program in programs:
            task_rel_counts[program] = programs[program] / task_sum
        output.loc[i] = [task_rel_counts]
        total_sum += task_sum
    return output, total_sum


# Computes shortest distance series from distance matrix
# NEGATIVE DISTANCE IN ORDER TO KEEP POSITIVE CORRELATIONS!!!
def get_shortest_distance(distance_matrix, negative=True):
    distances = pd.Series(index=distance_matrix.index)
    tasks = pd.Series(index=distance_matrix.index)
    distance_matrix.fillna(1000000, inplace=True)
    for i in distance_matrix.index:
        distance, task = min(distance_matrix.loc[i]), distance_matrix.loc[i].idxmin(axis=1)
        if negative:
            distances.loc[i] = -1 * distance
        else:
            distances.loc[i] = distance
        tasks.loc[i] = task
    return distances, tasks


def incorrect_evaluation(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path, task_sessions_path=task_sessions_path, tasks_path=tasks_path, task_sessions_cols=["id", "task"], tasks_cols=["id", "solution"])
    data = data[data.granularity == "execution"]
    print(data)
    data = data.fillna(False)
    data = data[data.new_correct != data.correct]
    print(data[["id", "correct", "new_correct"]])

    incorrect = data[data.new_correct == False]
    print(incorrect[["id", "task", "program", "solution", "correct", "new_correct"]])
    incorrect_non_51 = incorrect[data.task != 51]
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(incorrect_non_51[["id", "task", "program", "solution", "correct", "new_correct"]])

def last_with_empty_values(series):
    return series.iloc[-1]

def len_of_programs_dict(series):
    length = pd.Series(index=series.index)
    for i in series.index:
        length.loc[i] = len(series.loc[i][0])
    return length


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
    task_names_levels = {task[1].id: {"name": task[1].loc.name, "level": task[1].level} for task in tasks.iterrows()}
    return task_names_levels


# Computes median of lengths of a string series
def median_of_lens(series):
    return np.median([len(re.sub(pattern="[{}0123456789<>=!]", repl="", string=str(x))) for x in series])


# Draws 3D wireframe plot of frequent wrong programs ratio based on absolute and relative count of program occurences.
def plot_frequent_wrong_programs_ratio(tasks, total_sum, abs_step, abs_begin, abs_end, rel_step, rel_begin, rel_end, title=""):
    from mpl_toolkits.mplot3d import Axes3D

    abs_thresholds = [abs_step * i for i in range(abs_begin, abs_end)]
    rel_thresholds = [rel_step * i for i in range(rel_begin, rel_end)]
    frequents = [[] for _ in range(len(rel_thresholds))]
    for i, rel_threshold in enumerate(rel_thresholds):
        print(i)
        for abs_threshold in abs_thresholds:
            frequent = 0
            total_sum = 0
            for task in tasks.index.levels[0]:
                for seq in tasks.index:
                    if seq[0] == task:
                        this_seq = tasks.loc[seq]
                        if this_seq.abs_count >= abs_threshold and \
                                this_seq.rel_count >= rel_threshold and \
                                isinstance(this_seq.most_frequent_program, str):
                            frequent += this_seq.abs_count
                        if not total_sum:
                            task_total_sum = this_seq.task_freq
                total_sum += task_total_sum
                #print((frequent, total_sum))
            frequents[i].append(round(frequent / total_sum, 4))


    abs_axis = np.array([abs_thresholds for _ in range(len(rel_thresholds))])
    rel_axis = np.array([[item for _ in range(len(abs_thresholds))] for item in rel_thresholds])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(abs_axis, rel_axis, np.array(frequents), color="b")
    ax.set_xlabel("absolute count threshold")
    ax.set_ylabel("relative count threshold")
    ax.set_zlabel("frequent wrong programs ratio")
    if title:
        ax.set_title(title)
    plt.show()


# Replaces ambiguous "r" for "right" and "red" by "d" for "red", keeps "r" for "right"
# and saves the dataset
def replace_red_by_d(file_path, output_path, column_name):
    data = pd.read_csv(file_path)
    for i in data.index:
        text = data[column_name].loc[i]
        if isinstance(text, str):
            #print(data[column_name].loc[i])
            data[column_name].loc[i] = text.replace("r{", "d{")
    data.to_csv(output_path, index=False)


# Determines whether the sample solution is the most used correct solution
def sample_solution_not_most_frequent(solutions, programs):
    output = pd.Series(index=solutions.index)
    for i in solutions.index:
        #print(max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]))
        if solutions.loc[i] == max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]).replace("r{", "d{"):
            output.loc[i] = 0  #############
        else:
            #print(i, solutions.loc[i], max(programs.loc[i][0], key=lambda x: programs.loc[i][0][x]))
            output.loc[i] = 1  #############
    return output


def square_sequences_to_strings(sequence_series):
    print("Stringing")
    string_sequences = pd.Series(index=sequence_series.index)
    for i in sequence_series.index:
        seq = eval(sequence_series.loc[i])
        #if not np.isnan(sequence_series.loc[i]):
        #print(sequence_series.loc[i])
        string_square_sequence = ""
        for square in seq:
            for coord in square:
                string_square_sequence += str(coord)
        string_sequences.loc[i] = string_square_sequence
    return string_sequences


# Computes various statistics from mistake measures tasks
def statistics(tasks):
    tasks.rename(columns={"program": "absolute_counts"}, inplace=True)
    tasks["relative_counts"], total_sum = get_relative_counts(tasks.absolute_counts)
    tasks["total_wrong"] = pd.Series([sum(tasks.absolute_counts.loc[i][0].values()) for i in tasks.index], index=tasks.index)
    tasks["distinct_wrong"] = pd.Series([len(tasks.absolute_counts.loc[i][0]) for i in tasks.index], index=tasks.index)
    tasks["highest_abs_count"] = pd.Series([max(tasks.absolute_counts.loc[i][0].values()) for i in tasks.index], index=tasks.index)
    tasks["highest_rel_count"] = pd.Series([max(tasks.relative_counts.loc[i][0].values()) for i in tasks.index], index=tasks.index)
    return tasks, total_sum


def task_sessions_plot(task_sessions_path):
    ts = pd.read_csv(task_sessions_path)
    ts = ts.groupby("task").agg({"id": "count"})
    print(ts)
    ts.sort_values(by="id", ascending=False).plot.bar()
    plt.ylabel("number_of_task_sessions")
    plt.xlabel("tasks")
    plt.show()


#task_sessions_plot(task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv")
"""
replace_red_by_d(file_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 output_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 column_name="solution")
print("SNAPSHOTS")
replace_red_by_d(file_path="~/dp/Data/robomission-2018-11-03/program_snapshots.csv",
                 output_path="~/dp/Data/robomission-2018-11-03/program_snapshots_red_to_d.csv",
                 column_name="program")

"""
"""
load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots_extended.csv",
                        task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                        task_sessions_cols=None)
load_task_names_levels(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks_red_to_d.csv")
"""
"""
synchronous_interpreter_correctness_and_square_sequence(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_red_to_d.csv",
                                                        task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                                                        tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                                                        output_snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv")
"""
"""
incorrect_evaluation(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                     task_sessions_path="~/dp//Data/robomission-2018-11-03/task_sessions.csv",
                     tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv")
"""
