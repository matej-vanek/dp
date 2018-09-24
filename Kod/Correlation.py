import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from Tools import *


# Computes task difficulty dataframe
# Creates task dataframe of successfulness of sessions, successfulness of submits and number of block types
def difficulty_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "solved", "time_spent"],
                                   tasks_cols=["id", "solution", "mission"])

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "solved": last,
                                                     "granularity": count_submits,
                                                     "correct": count_true,
                                                     "solution": last})

    # successfulness of sessions
    successful_sessions = all_sessions.loc[all_sessions["solved"]]
    all_sessions_by_tasks = all_sessions.groupby("task").agg({"solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"solved": "count"})

    difficulty = successful_sessions_by_tasks / all_sessions_by_tasks
    difficulty.rename(columns={"solved": "task_sessions_solved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    # successfulness of submits
    submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "correct": "sum"})
    difficulty["submits_correct"] = submits_by_tasks["correct"] / submits_by_tasks["granularity"]

    del submits_by_tasks

    # number of block types
    block_types_by_task = all_sessions.groupby("task").agg({"solution": last})
    difficulty["block_types"] = count_distinct_blocks(block_types_by_task["solution"], 1)
    difficulty["block_types"] = difficulty["block_types"].astype("int64")

    difficulty["block_types_flr"] = count_distinct_blocks(block_types_by_task["solution"], 3)
    difficulty["block_types_flr"] = difficulty["block_types_flr"].astype("int64")

    difficulty["block_types_flrs"] = count_distinct_blocks(block_types_by_task["solution"], 4)
    difficulty["block_types_flrs"] = difficulty["block_types_flrs"].astype("int64")

    print(difficulty)
    return difficulty


# Computes task difficulty dataframe
# Creates task dataframe of median time, median edits, median submits, median solution length, sample solution length,
# deletion ratio, median deletions
def complexity_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "solved", "time_spent"],
                                   tasks_cols=["id", "solution", "mission"])
    data = data[data["solved"]]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "spent_time": "max",
                                                     "granularity": [count_edits, count_submits],
                                                     "program": [last,
                                                                 count_deletions(consider_multideletions=False), # TODO: partial application
                                                                 count_deletions(consider_multideletions=True)], # TODO: partial application
                                                     "solution": last},
                                                     # TODO:??deletions??
                                                    )

complexity_measures(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                    task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                    tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")

# Computes correlation of task measures and creates heat table
def tasks_measures_correlations(task_table, method, title):
    correlations = task_table.corr(method=method)
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    return correlations
# TODO: think about how to perform by_levels correlation


# Computes correlation of correlation methods and creates heat table
# BE AWARE of difference between cerrelation of FULL CORRELATION TABLES and of REDUCED (triangle) CORRELATION TABLES!!!
def correlation_methods_correlations(pearson_tasks_measures_correlation, spearman_tasks_measures_correlation,
                                     variable_group_title, full_or_triangle):
    if full_or_triangle == "full":
        correlations = np.corrcoef(np.ndarray.flatten(pearson_tasks_measures_correlation.as_matrix()),
                                   np.ndarray.flatten(spearman_tasks_measures_correlation.as_matrix()))
    elif full_or_triangle == "triangle":
        correlations = np.corrcoef(flattened_triangle_table(pearson_tasks_measures_correlation.as_matrix()),
                                   flattened_triangle_table(spearman_tasks_measures_correlation.as_matrix()))
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title("""
    Pearson {}-matrix correlation of Pearson and Spearman correlation methods\n
    applied to {}
    """.format(full_or_triangle, variable_group_title))
    plt.show()

    return correlations


def full_and_triangle_correlation(corr_of_full_corr_tables, corr_of_triangle_corr_tables, variable_group_title):
    correlations = np.corrcoef(np.ndarray.flatten(np.array(corr_of_full_corr_tables)),
                               np.ndarray.flatten(np.array(corr_of_triangle_corr_tables)))
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title("""
    Pearson correlation of full and triangle method\n
    applied to {}
    """.format(variable_group_title))
    plt.show()

    return correlations


def all_correlations(snapshots_path, task_sessions_path, tasks_path, measures_function, variable_group_title):
    tasks_measures_table = measures_function(snapshots_path=snapshots_path,
                                             task_sessions_path=task_sessions_path,
                                             tasks_path=tasks_path)
    pearson_tasks_measures_correlation = tasks_measures_correlations(task_table=tasks_measures_table,
                                                                     method="pearson",
                                                                     title="Pearson correlation of {}"
                                                                     .format(variable_group_title))
    spearman_tasks_measures_correlation = tasks_measures_correlations(task_table=tasks_measures_table,
                                                                      method="spearman",
                                                                      title="Spearman correlation of {}"
                                                                      .format(variable_group_title))
    corr_of_full_corr_tables = \
        correlation_methods_correlations(pearson_tasks_measures_correlation=pearson_tasks_measures_correlation,
                                         spearman_tasks_measures_correlation=spearman_tasks_measures_correlation,
                                         variable_group_title=variable_group_title,
                                         full_or_triangle="full")
    corr_of_triangle_corr_tables = \
        correlation_methods_correlations(pearson_tasks_measures_correlation=pearson_tasks_measures_correlation,
                                         spearman_tasks_measures_correlation=spearman_tasks_measures_correlation,
                                         variable_group_title=variable_group_title,
                                         full_or_triangle="triangle")

    full_and_triangle_correlation(corr_of_full_corr_tables=corr_of_full_corr_tables,
                                  corr_of_triangle_corr_tables=corr_of_triangle_corr_tables,
                                  variable_group_title=variable_group_title)


"""
all_correlations(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=difficulty_measures,
                 variable_group_title="difficulty measures")
"""
