#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matej Vanek's RoboMission dashboard creator.
"""

"""
File name: dashboard_creator.py
Author: Matej Vanek
Created: 2018-12-
Python Version: 3.6
"""

import editdistance
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from functools import partial

from tools import count_similar_tasks, dict_of_counts, flatten_table_remove_nan, get_shortest_distance, \
    load_extended_snapshots, square_sequences_to_strings


def parse():
    """
    Parses arguments from command line.
    :return: dict; dictionary of <argument: value> pairs
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--snapshots_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission program snapshots")
    parser.add_argument('-ts', '--task_sessions_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission task sessions")
    parser.add_argument('-t', '--tasks_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission tasks")
    args = vars(parser.parse_args())
    return args



# tasks: task_id, section, sample_solution, (num_of_ts), (num_of_learners),
#        (conflict_solutions), DIFFICULTY:median_correct_time, dict_of_all_correct_solutions, SIMILARITY:levenshtein_5,
#        PROBLEMS:dict_of_leaving+dict_of_wrong_submissions
# learners: TS_PERF:time, TOTAL_PERF:solved_tasks
def compute(args):
    data = load_extended_snapshots(snapshots_path=args["snapshots_path"],
                                   task_sessions_path=args["task_sessions_path"],
                                   tasks_path=args["tasks_path"],
                                   task_sessions_cols=["id", "solved", "time_spent", "task"],
                                   tasks_cols=["id", "section", "solution", ])

    data.correct = data.correct.fillna(False)
    data.new_correct = data.new_correct.fillna(False)
    data = data[data.new_correct == data.correct]

    tasks = data.groupby("task").agg({"section": "last",
                                      "solution": "last",
                                      "time_spent": "median",

                                      })

    levenshtein_matrix = pd.DataFrame(data=None, index=sorted(tasks.index), columns=sorted(tasks.index))
    for i in levenshtein_matrix.index:
        for j in levenshtein_matrix.index:
            if i < j:
                levenshtein_matrix.loc[i][j] = editdistance.eval(tasks.solution.loc[i], tasks.solution.loc[j])
    flat_levenshtein_matrix = flatten_table_remove_nan(levenshtein_matrix)
    for i in levenshtein_matrix.index:
        for j in levenshtein_matrix.index:
            if i > j:
                levenshtein_matrix.loc[i][j] = levenshtein_matrix.loc[j][i]
    tasks["levenshtein10"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.1))
    tasks["closest_distance"], tasks["closest_task"] = get_shortest_distance(levenshtein_matrix, negative=False)

    programs = data[data.granularity == "execution"][data.new_correct]
    programs.square_sequence = square_sequences_to_strings(programs.square_sequence)
    programs = programs.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})

    programs["representant"] = pd.Series([max(programs.loc[i].program[0], key=programs.loc[i].program[0].get) for i in programs.index], index=programs.index)
    programs["occurences"] = pd.Series([sum([programs.loc[i].program[0][solution] for solution in programs.loc[i].program[0]]) for i in programs.index], index=programs.index)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        pd.set_option('display.width', 700)
        print(tasks)
        print(programs)

    #PROBLEMS:dict_of_leaving+dict_of_wrong_submissions


def visualize():
    pass


if __name__ == '__main__':
    # IF RED-D AND COMPUTE SEQUENCES:...
    args = parse()
    results = compute(args)
    visualize(results)