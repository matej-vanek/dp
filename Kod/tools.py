#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast import literal_eval
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.cluster.hierarchy import fcluster, linkage

from robomission_ast import *
from minirobocode_interpreter import run_task


def bag_of_blocks(series):
    """
    Counts bag-of-blocks characteristics of tasks.
    :param series: pd.Series; solutions series
    :return: pd.Series; bag-of-blocks vectors series
    """
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


def bag_of_entities(task_setting_series):
    """
    Counts bag-of-task_game_world_entities.
    :param task_setting_series: pd.Series; task settineg series
    :return: pd.Series; bag-of-entities vectors series
    """
    bag = []
    for i in task_setting_series.index:
        task = literal_eval(task_setting_series.loc[i])
        task = re.sub("r", "d", task["fields"])

        entities = [0 for _ in range(6)]  # [all_squares, colorful_squares, diamonds, wormholes, asteroids, meteoroids]
        entities[0] = len(re.findall("[bkygd]", task))
        entities[1] = len(re.findall("[ygd]", task))
        entities[2] = len(re.findall("D", task))
        entities[3] = len(re.findall("[WXYZ]", task))
        entities[4] = len(re.findall("A", task))
        entities[5] = len(re.findall("M", task))
        bag.append(entities)
    bag = pd.Series(bag, index=task_setting_series.index)
    return bag


def count_deletions(series, mode):
    """
    Counts deletions in code.
    :param series: pd.Series; program series
    :param mode: string; if "all"   -> difference of lengths of strings
                         if "edits" -> number of program-shortening edits
                         if "1_0"   -> binary flag of any program shortening in task session
    :return: pd.Series; deletions series
    """
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
            elif mode == "edits":
                dels += 1
            elif mode == "1_0":
                dels = 1
        last = item
    return dels


def count_distinct_blocks(series, basic_block_types_number):
    """
    Counts distinct blocks used in MiniRoboCode programs.
    :param series: pd.Series; series of MiniRoboCode programs
    :param basic_block_types_number: int; determines how many of "f", "l", "r" and "s" blocks collapse into the only one
    :return: pd.Series; series of counts
    """
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
    """
    Counts edit operations.
    :param series: pd.Series; 'granularity' series
    :return: int; number of edit operations
    """
    count = 0
    for item in series:
        if item == "edit":
            count += 1
    return count


def count_frequent_wrong_programs_ratio(tasks, abs_threshold, rel_threshold):
    """
    Counts ratio of frequent wrong programs to all wrong programs.
    :param tasks: pd.DataFrame; tasks DataFrame
    :param abs_threshold: number; minimal threshold on the minimal absolute frequency of a problem to be frequent
    :param rel_threshold: number; minimal threshold on the minimal relative frequency of a problem to be frequent
    :return: pd.Series, pd.Series, pd.Series; ratio series, number of unique programs series,
    all programs + their absolute frequency series
    """
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


def count_program_clusters(programs):
    """
    Counts number of program clusters of tasks.
    Builds ASTs from programs, computes their TED matrix, hierarchically clusters them,
    prunes where cophenetic dist is > 5, returns number of clusters.
    :param programs: pd.Series of {string: int} dicts; programs frequency series
    :return: pd.Series, dict, dict; number of clusters series, info about each program, info about each cluster
    """
    clusters_count = pd.Series(index=programs.index)
    program_info = {}
    cluster_info = {}
    for task in programs.index:
        program_list = list(programs.loc[task][0].keys())
        if len(programs.loc[task][0].keys()) > 1:
            condensed_dist_matrix = []
            program_ast_list = np.array(list(map(partial(build_ast), program_list)))
            for i in range(len(program_ast_list)):
                for j in range(len(program_ast_list)):
                    if i < j:
                        condensed_dist_matrix.append(ast_ted(program_ast_list[i], program_ast_list[j]))
            condensed_dist_matrix = np.ndarray.flatten(np.array(condensed_dist_matrix))
        else:
            condensed_dist_matrix = [0]
        cluster_assign = fcluster(linkage(condensed_dist_matrix), 5, criterion="distance")

        program_info[task] = {program: {'cluster': cluster, 'freq': programs.loc[task][0][program]}
                              for program, cluster in zip(program_list, cluster_assign)}

        cluster_info[task] = {cluster: {'programs': [program for program in program_info[task]
                                                     if program_info[task][program]["cluster"] == cluster],
                                        'representative': max([prog for prog in program_info[task]
                                                               if program_info[task][prog]["cluster"] == cluster],
                                                              key=lambda x: program_info[task][x]["freq"])}
                              for cluster in set(cluster_assign)}
        #print(cluster_info[task])
        clusters_count.loc[task] = len(set(cluster_assign))
    return clusters_count, program_info, cluster_info


def count_similar_tasks(distance_matrix, threshold):
    """
    Counts how many tasks in the distance matrix have lower distance to the source task than threshold. Computes this
    for all rows of the distance matrix.
    :param distance_matrix: pd.DataFrame; distance matrix
    :param threshold: number; the upper threshold determining whether tasks are similar or not
    :return: pd.Series; series of the number of similar tasks
    """
    output = pd.Series(index=distance_matrix.index)
    for i in distance_matrix.index:
        x = [1 for task in distance_matrix if distance_matrix.loc[i, task] <= threshold]
        output.loc[i] = sum(x)
    return output


def count_submits(series):
    """
    Counts submit operations.
    :param series: pd.Series; 'granularity' series
    :return: int; number of submit operations
    """
    count = 0
    for item in series:
        if item == "execution":
            count += 1
    return count


def count_task_frequency(dataframe):
    """
    Counts absolute frequency of task in two-level index dataframe
    :param dataframe: pd.DataFrame; two-level index dataframe, task is the first level of index,
    dataframe has column 'abs_freq'
    :return: pd.Series; series of task absolute frequencies
    """
    task_freq_series = pd.Series(index=dataframe.index)
    for i in dataframe.index.levels[0]:
        task_freq = 0
        for j in dataframe.index:
            if j[0] == i:
                task_freq += dataframe.loc[j].abs_count
        for j in dataframe.index:
            if j[0] == i:
                task_freq_series.loc[j] = task_freq
    return task_freq_series


def count_total_abs_freq(series):
    """
    Counts total absolute frequency of a frequency series.
    :param series: pd.Series; frequency series
    :return: pd.Series; total absolute frequency series
    """
    abs_freq = pd.Series(index=series.index)
    for i in series.index:
        abs_freq.loc[i] = sum([series.loc[i][0][key] for key in series.loc[i][0]])
    return abs_freq


def count_true(series):
    """
    Counts True values.
    :param series: pd.Series; series of boolean values
    :return: pd.Series; number of True values
    """
    count = 0
    for item in series:
        if item is True:
            count += 1
    return count



# counts all block types used in all items of series
def count_used_blocks(series):
    """
    Counts distinct blocks in series (aggregating function).
    :param series: pd.Series; series of programs
    :return: number of distinct blocks
    """
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



def dict_of_counts(series, del_false=False):
    """
    Counts occurrences of items in series.
    :param series: pd.Series; series of items to count
    :param del_false: bool; if True, items equal to False are deleted from the result
    :return: dict in list; dict of items and numbers of their occurrences
    """
    solutions = {}
    for item in series:
        if item not in solutions:
            solutions[item] = 0
        solutions[item] += 1
    if del_false:
        if False in solutions:
            del solutions[False]
    return [solutions]


def entropy(occurrence_dict):
    """
    Counts n-ary entropy of items.
    :param occurrence_dict: dict; item: relative frequencies of items dictionary
    :return: n-ary entropy of items
    """
    if len(occurrence_dict[0]) == 1:
        return 0
    occurrence_list = occurrence_dict[0].values()
    frequency_list = [i/sum(occurrence_list) for i in occurrence_list]
    return 1/np.log2(len(frequency_list)) * sum([- x * np.log2(x) for x in frequency_list])


def flatten_table_remove_nan(table, triangle=False):
    """
    Transforms table to one-row representation, deletes None values,
    :param table: pd.DataFrame; table to be flattened
    :param triangle: bool; if True, it omits all values above the main diagonal
    :return: list; flattened table
    """
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


def flattened_triangle_table_from_array(table):
    """
    Transforms table to one-row representation, deletes None values,
    :param table: np.array; table to be flattened
    :return: list; flattened table
        """
    reduced_table = []
    for i in range(len(table)):
        for j in range(i):
            reduced_table.append(table[i][j])
    return reduced_table


def get_most_frequent_program(program_series):
    """
    Returns the most frequent program for each row of series
    :param program_series: pd.Series; series of dicts in list, [{program: frequency}]
    :return: pd.Series; series of most frequent programs
    """
    most_freq_program = pd.Series(index=program_series.index)
    for i in program_series.index:
        most_freq_program.loc[i] = max(program_series.loc[i][0], key=lambda x: program_series.loc[i][0][x])
    return most_freq_program


def get_shortest_distance(distance_matrix, negative=True):
    """
    Finds shortest distance in each row of matrix. Distances returns implicitly AS NEGATIVE NUMBERS in order to keep
    correlations with other measures positive
    :param distance_matrix: pd.DataFrame; distance matrix
    :param negative: bool if True, distances are transformed to negative numbers
    :return: pd.Series, pd.Series; shortest distances series, closest tasks series
    """
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


def last_with_empty_values(series):
    """
    Returns last value of the series, regardless whether it is empty or not.
    :param series: pd.Series; series
    :return: ???; last value
    """
    return series.iloc[-1]


def len_of_last(series):
    """
    Returns length of the last item of series.
    :param series:
    :return: int; length of last item
    """
    return len(re.sub("[{}0123456789<>=!]", "", series.iloc[-1]))


def len_of_programs_dict(series):
    length = pd.Series(index=series.index)
    for i in series.index:
        length.loc[i] = len(series.loc[i][0])
    return length


def load_extended_snapshots(snapshots_path, task_sessions_path, tasks_path, task_sessions_cols, tasks_cols):
    """
    Merges snapshots, task_sessions and tasks together (left outer join) and returns the result.
    :param snapshots_path: string; path to .csv file with RoboMission program snapshots
    :param task_sessions_path: string; path to .csv file with RoboMission task sessions
    :param tasks_path: string; path to .csv file with RoboMission tasks
    :param task_sessions_cols: list of strings; list of task-sessions columns to be merged to snapshots
    :param tasks_cols: list of strings; list of tasks columns to be merged to snapshots
    :return: pd.DataFrame; merged dataframe
    """
    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=task_sessions_cols)
    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)
    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")

    if tasks_cols:
        tasks = pd.read_csv(tasks_path, usecols=tasks_cols)
        tasks.rename(index=str, columns={"id": "task"}, inplace=True)
        snapshots_with_tasks = pd.merge(snapshots_with_tasks, tasks, how="left", on="task")
    return snapshots_with_tasks


def median_of_lens(series):
    """
    Computes median of lengths of a string series.
    :param series: pd.Series; series of strings
    :return: float; median of string lengths
    """
    return np.median([len(re.sub(pattern="[{}0123456789<>=!]", repl="", string=str(x))) for x in series])


def plot_frequent_wrong_programs_ratio(tasks, abs_step, abs_begin, abs_end, rel_step, rel_begin, rel_end):
    """
    Creates plots of frequent wrong programs ratio based on thresholds.
    :param tasks: string; path to .csv file with RoboMission tasks
    :param abs_step: number; interval between two absolute thresholds values
    :param abs_begin: number; first value of abs_step to be computed
    :param abs_end: number; last value of abs_step to be computed
    :param rel_step: number; interval between two relative thresholds values
    :param rel_begin: number; first value of rel_step to be computed
    :param rel_end: number; last value of rel_step to be computed
    :return:
    """
    abs_thresholds = [round(abs_step * i, 2) for i in range(abs_begin, abs_end)]
    rel_thresholds = [round(rel_step * i, 2) for i in range(rel_begin, rel_end)]

    frequents = pd.DataFrame(index=rel_thresholds, columns=abs_thresholds)
    for rel_threshold in rel_thresholds:
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
            frequents.loc[rel_threshold][abs_threshold] = round(frequent / total_sum, 4)
    frequents.index.name = "relative_threshold"
    print(frequents)

    colors = ['darkred', 'red', 'pink', 'orange', 'yellow', 'lawngreen', 'cyan', 'dodgerblue', 'navy', 'black']

    for i in range(len(frequents.index)):
        plt.plot(frequents.index, frequents.iloc[:, [i]], marker='', color=colors[i], linewidth=2)
    plt.legend(title="absolute count threshold", labels=abs_thresholds)
    plt.xlabel("relative count threshold")
    plt.ylabel("frequent wrong programs ratio")
    plt.show()

    for i in range(len(frequents.iloc[0])):
        plt.plot(list(frequents), frequents.iloc[i], marker='', color=colors[i], linewidth=2)
    plt.legend(title="relative count threshold", labels=rel_thresholds)
    plt.xlabel("absolute count threshold")
    plt.ylabel("frequent wrong programs ratio")
    plt.show()


def replace_red_by_d(file_path, output_path, column_names):
    """
    Replaces ambiguous "r" for "right" and "red" by "d" for "red", keeps "r" for "right" and saves the dataset.
    :param file_path: string; path to .csv file to be processed
    :param output_path: string; path to folder where modified file will be saved
    :param column_names: list of strings; list of column names to be processed
    :return: pd.DataFrame; file with replaced "red" symbol
    """
    data = pd.read_csv(file_path)
    for i in data.index:
        for column_name in column_names:
            text = data[column_name].loc[i]
            if isinstance(text, str):
                data[column_name].loc[i] = text.replace("r{", "d{")
    data.to_csv(output_path, index=False)
    return data


def square_sequences_to_strings(sequence_series):
    """
    Transforms square sequences to string representation.
    :param sequence_series: pd.Series; series of square sequences
    :return: pd.Series; series of square sequences string representation
    """
    string_sequences = pd.Series(index=sequence_series.index)
    for i in sequence_series.index:
        seq = eval(sequence_series.loc[i])
        string_square_sequence = ""
        for square in seq:
            for coord in square:
                string_square_sequence += str(coord)
        string_sequences.loc[i] = string_square_sequence
    return string_sequences


def statistics(series):
    """
    Returns 0.1, 0.5 and 0.9 -th quantile and number of unique values of series
    :param series: pd.Series of numbers; series with values to be processed
    :return: [float, float, float, int]; 0.1, 0.5 and 0.9 -th quantile, number of unique values
    """
    return [series.quantile(0.1), series.quantile(0.5), series.quantile(0.9), series.nunique(dropna=True)]


def synchronous_interpreter_run(snapshots_path=None, task_sessions_path=None, tasks_path=None,
                                output_snapshots_path=None, only_executions=True, only_edits=False,
                                data_frame=None, save=True):
    """
    Reruns submits in data with synchronous interpreter, computes new information about program correctness and visited
    squares sequences.
    :param snapshots_path: string; path to snapshots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :param output_snapshots_path: string; output path to new snapshots .csv file
    :param only_executions: bool; if True, only executions records are rerun
    :param only_edits: bool; if True, only edits records are rerun
    :param data_frame: pd.DataFrame; if present, data load from data_frame instead of file paths
    :param save: bool; if True, extended snapshots .csv file with new correctness and visited squares sequences is saved
    :return: pd.DataFrame; extended snapshots .csv file with new correctness and visited squares sequences
    """
    if data_frame is not None:
        data = data_frame
    else:
        data = load_extended_snapshots(snapshots_path=snapshots_path,
                                       task_sessions_path=task_sessions_path,
                                       tasks_path=tasks_path,
                                       task_sessions_cols=["id", "task"],
                                       tasks_cols=[])
        data["new_correct"] = pd.Series(None, index=data.index)
        data["square_sequence"] = pd.Series(None, index=data.index)
    for i in data.index:
        if only_executions:
            if data.loc[i].granularity != "execution":
                continue
        if only_edits:
            if data.loc[i].granularity != "edit":
                continue
        correct, square_sequence = run_task(tasks_path=tasks_path,
                                            task_id=data.loc[i].task,
                                            program=data.loc[i].program,
                                            verbose=False)
        data.new_correct.loc[i] = str(correct)
        data.square_sequence.loc[i] = str(square_sequence)
    if save:
        data = data.drop(["task"], axis=1)
        data.to_csv(output_snapshots_path, index=False)
    return data


def task_ids_to_phases(tasks_path, task_ids):
    """
    Translates task-ids to phase value
    :param tasks_path: string; path to .csv file with RoboMission tasks
    :param task_ids: pd.Series; task ids series
    :return: pd.Series; phases series
    """
    tasks = pd.read_csv(tasks_path)
    phases = []
    for task_id in task_ids:
        phases.extend(list(tasks[tasks.id == task_id].section))
    return phases






#print(task_ids_to_phases("/home/matejvanek/dp/Data/robomission-2018-11-03/tasks.csv", [11, 3, 51, 83, 14]))

"""
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
        #                           'representative': max(program_info[task][0], key=program_info[task][0].get)}
        #                           for cluster in set(cluster_assign)}

        print(set(cluster_assign))
        for cluster in set(cluster_assign):
            print(cluster)
            print(program_info[task])
            programs = [program_info[task][program] for program in program_info[task] if
                                                program_info[task][program]["cluster"] == cluster]
            representative = max(program_info[task][0], key=program_info[task][0].get)

            cluster_info[task] = {cluster: {"programs": programs, "representative": representative}}

        #cluster_info[task] = {cluster:
        #                          {'programs': [program_info[task][program] for program in program_info[task] if
        #                                        program_info[task][program]["cluster"] == cluster],
        #                           'representative': max(program_info[task][0], key=program_info[task][0].get)}
        #                      for cluster in set(cluster_assign)}



        #print(cluster_assign)
        #print("Number of found clusters: ", len(set(cluster_assign)))
        clusters_count.loc[task] = len(set(cluster_assign))
        print(program_info[task])
        print(cluster_info)
        print(cluster_info[task])
    return clusters_count, program_info, cluster_info
"""


#task_sessions_plot(task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv")
"""
replace_red_by_d(file_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 output_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 column_name="solution")
print("SNAPSHOTS")
replace_red_by_d(file_path="~/dp/Data/robomission-2018-11-03/program_snapshots_qqq.csv",
                 output_path="~/dp/Data/robomission-2018-11-03/program_snapshots_qqq_red_to_d.csv",
                 column_names=["program"])
"""
"""
load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots_extended.csv",
                        task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                        task_sessions_cols=None)
load_task_names_levels(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks_red_to_d.csv")
"""
"""
synchronous_interpreter_run(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_qqq_red_to_d.csv",
                                                        task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                                                        tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                                                        output_snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_qqq_extended.csv")
"""
"""
incorrect_evaluation(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                     task_sessions_path="~/dp//Data/robomission-2018-11-03/task_sessions.csv",
                     tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv")
"""
