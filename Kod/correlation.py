#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import editdistance
import seaborn as sns
from scipy.spatial.distance import euclidean

from tools import *


def difficulty_and_complexity_measures(snapshots_path, task_sessions_path, tasks_path):
    """
    Computes difficulty and complexity measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; difficulty and complexity measures
    """
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "solved", "time_spent"],
                                   tasks_cols=["id", "solution"])
    data = data.fillna(False)
    data = data[data.correct == data.new_correct]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": last_with_empty_values})
    all_sessions["new_solved"] = all_sessions.new_correct.astype(bool)

    successful_sessions = all_sessions[all_sessions.new_solved]
    all_sessions_by_tasks = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"new_solved": "count"})

    difficulty_and_complexity = 1 - successful_sessions_by_tasks / all_sessions_by_tasks
    difficulty_and_complexity.rename(columns={"new_solved": "task_sessions_unsolved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    distinct_blocks_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty_and_complexity["distinct_blocks_1"] = count_distinct_blocks(distinct_blocks_by_task.solution, 1)
    difficulty_and_complexity.distinct_blocks_1 = difficulty_and_complexity.distinct_blocks_1.astype("int64")

    difficulty_and_complexity["distinct_blocks_3"] = count_distinct_blocks(distinct_blocks_by_task.solution, 3)
    difficulty_and_complexity.distinct_blocks_3 = difficulty_and_complexity.distinct_blocks_3.astype("int64")

    difficulty_and_complexity["distinct_blocks_4"] = count_distinct_blocks(distinct_blocks_by_task.solution, 4)
    difficulty_and_complexity.distinct_blocks_4 = difficulty_and_complexity.distinct_blocks_4.astype("int64")

    data["granularity_submits"] = data.granularity
    data["program_all"] = data.program
    data["program_edits"] = data.program
    data["program_1_0"] = data.program

    task_sessions = data.groupby("task_session").agg({"task": "last",
                                                      "time_spent": "max",
                                                      "solution": "last",
                                                      "granularity": count_edits,
                                                      "granularity_submits": count_submits,
                                                      "program": last_with_empty_values,
                                                      "program_all": partial(count_deletions, mode="all"),
                                                      "program_edits": partial(count_deletions, mode="edits"),
                                                      "program_1_0": partial(count_deletions, mode="1_0")})

    tasks = task_sessions.groupby("task").agg({"time_spent": "median",
                                               "granularity": "median",
                                               "granularity_submits": "median",
                                               "program": median_of_lens,
                                               "solution": len_of_last,
                                               "program_all": "median",
                                               "program_edits": "median",
                                               "program_1_0": ["median", "sum", "count"]})
    tasks.deletion_ratio = tasks[("program_1_0", "sum")] / tasks[("program_1_0", "count")]

    difficulty_and_complexity["median_time"] = tasks[("time_spent", "median")]
    difficulty_and_complexity["median_edits"] = tasks[("granularity", "median")]
    difficulty_and_complexity["median_submissions"] = tasks[("granularity_submits", "median")]
    difficulty_and_complexity["median_solution_length"] = tasks[("program", "median_of_lens")]
    difficulty_and_complexity["sample_solution_length"] = tasks[("solution", "len_of_last")]
    difficulty_and_complexity["deletion_ratio"] = tasks.deletion_ratio
    difficulty_and_complexity["median_deletions_all"] = tasks[("program_all", "median")]
    difficulty_and_complexity["median_deletions_edits"] = tasks[("program_edits", "median")]
    difficulty_and_complexity["median_deletions_1_0"] = tasks[("program_1_0", "median")]

    for column in difficulty_and_complexity:
        print(column, statistics(difficulty_and_complexity[column]))

    return difficulty_and_complexity


def solution_uniqueness_measures(snapshots_path, task_sessions_path, tasks_path):
    """
    Computes solution uniqueness measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; solution uniqueness measures
    """
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "task", "solved"],
                                   tasks_cols=["id", "solution"])
    data = data[data.new_correct.notnull()]
    data = data[data.new_correct]
    data = data[data.new_correct == data.correct]

    tasks = data.groupby("task").agg({"program": dict_of_counts,
                                      "square_sequence": dict_of_counts,
                                      "solution": "last"})

    uniqueness = pd.DataFrame(index=tasks.index)
    uniqueness["solutions_entropy"] = list(map(entropy, tasks.program))
    uniqueness["squares_sequences_entropy"] = list(map(entropy, tasks.square_sequence))
    tasks["distinct_solutions"] = [len(x[0]) for x in tasks.program]
    tasks["distinct_squares_sequences"] = [len(x[0]) for x in tasks.square_sequence]
    uniqueness["unique_solutions"] = tasks.distinct_solutions
    uniqueness["unique_squares_sequences"] = tasks.distinct_squares_sequences
    uniqueness["program_clusters_count"], _, _ = count_program_clusters(tasks.program)

    for column in uniqueness:
        print(column, statistics(uniqueness[column]))

    return uniqueness


def task_similarity_measures(snapshots_path, task_sessions_path, tasks_path):
    """
    Computes similarity measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; similarity measures
    """
    del snapshots_path
    del task_sessions_path

    tasks = pd.read_csv(tasks_path, index_col="id")
    sample_solutions = tasks.solution
    asts = pd.Series(list(map(build_ast, sample_solutions)), index=sample_solutions.index)
    bags_of_blocks = bag_of_blocks(sample_solutions)
    bags_of_entities = bag_of_entities(tasks.setting)

    ast_ted_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index),
                                  columns=sorted(sample_solutions.index))
    levenshtein_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index),
                                      columns=sorted(sample_solutions.index))
    bag_of_blocks_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index),
                                        columns=sorted(sample_solutions.index))
    bag_of_entities_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index),
                                          columns=sorted(sample_solutions.index))

    for i in sorted(sample_solutions.index):
        for j in sorted(sample_solutions.index):
            if i < j:
                ast_ted_matrix.loc[i][j] = ast_ted(asts.loc[i], asts.loc[j])
                levenshtein_matrix.loc[i][j] = editdistance.eval(sample_solutions.loc[i], sample_solutions.loc[j])
                bag_of_blocks_matrix.loc[i][j] = euclidean(bags_of_blocks.loc[i], bags_of_blocks.loc[j])
                bag_of_entities_matrix.loc[i][j] = euclidean(bags_of_entities.loc[i], bags_of_entities.loc[j])

    matrices = [ast_ted_matrix, levenshtein_matrix, bag_of_blocks_matrix, bag_of_entities_matrix]
    titles = ["AST TED", "Levenshtein", "Bag-of-blocks", "Bag-of-entities"]
    for i in range(len(matrices)):
        frequencies = dict(Counter(matrices[i].values.flatten()))
        plt.bar(list(frequencies.keys()), list(frequencies.values()), width=0.05, color='g')
        plt.title(titles[i] + " distances distribution")
        plt.xlabel("distance")
        plt.ylabel("count")
        plt.show()

    flat_ast_ted_matrix = flatten_table_remove_nan(ast_ted_matrix)
    flat_levenshtein_matrix = flatten_table_remove_nan(levenshtein_matrix)
    flat_bag_of_blocks_matrix = flatten_table_remove_nan(bag_of_blocks_matrix)
    flat_bag_of_entities_matrix = flatten_table_remove_nan(bag_of_entities_matrix)

    for i in sorted(sample_solutions.index):
        for j in sorted(sample_solutions.index):
            if i > j:
                ast_ted_matrix.loc[i][j] = ast_ted_matrix.loc[j][i]
                levenshtein_matrix.loc[i][j] = levenshtein_matrix.loc[j][i]
                bag_of_blocks_matrix.loc[i][j] = bag_of_blocks_matrix.loc[j][i]
                bag_of_entities_matrix.loc[i][j] = bag_of_entities_matrix.loc[j][i]

    similarity = pd.DataFrame(index=tasks.index)
    similarity["ast_ted2"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.02))
    similarity["ast_ted5"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.05))
    similarity["ast_ted10"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.10))
    similarity["levenshtein2"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.02))
    similarity["levenshtein5"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.05))
    similarity["levenshtein10"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.10))
    similarity["blocks2"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.02))
    similarity["blocks5"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.05))
    similarity["blocks10"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.10))
    similarity["entities2"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.02))
    similarity["entities5"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.05))
    similarity["entities10"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.10))
    similarity["closest_ast"], _ = get_shortest_distance(ast_ted_matrix)
    similarity["closest_levenshtein"], _ = get_shortest_distance(levenshtein_matrix)
    similarity["closest_blocks"], _ = get_shortest_distance(bag_of_blocks_matrix)
    similarity["closest_entities"], _ = get_shortest_distance(bag_of_entities_matrix)

    for column in similarity:
        print(column, statistics(similarity[column]))
    return similarity


def frequent_problems_measures(snapshots_path, task_sessions_path, tasks_path, **kwargs):
    """
    Computes frequent problems measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; frequent problems measures
    """
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "task"],
                                   tasks_cols=[])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]

    last_ts_snapshots = data.groupby("task_session").agg({"task": "max",
                                                          "new_correct": count_true,
                                                          "granularity": "last",
                                                          "program": last_with_empty_values,
                                                          "square_sequence": last_with_empty_values})

    last_ts_snapshots = last_ts_snapshots[[isinstance(x, str) for x in last_ts_snapshots.program]]
    last_ts_snapshots["new_solved"] = last_ts_snapshots.new_correct.astype(bool)

    wrong_ts = last_ts_snapshots[last_ts_snapshots.new_solved == 0]
    wrong_ts = synchronous_interpreter_run(data_frame=wrong_ts,
                                           only_executions=False,
                                           only_edits=True,
                                           save=False,
                                           tasks_path=tasks_path)
    wrong_ts["string_square_sequence"] = square_sequences_to_strings(wrong_ts.square_sequence)

    del last_ts_snapshots
    tasks_left = wrong_ts.groupby(["task", "string_square_sequence"]).agg(
        {"program": partial(dict_of_counts, del_false=True), "new_solved": "count"})
    del wrong_ts
    tasks_left["distinct_programs"] = len_of_programs_dict(tasks_left.program)
    tasks_left["most_frequent_program"] = get_most_frequent_program(tasks_left.program)
    tasks_left["abs_count"] = count_total_abs_freq(tasks_left.program)
    tasks_left["task_freq"] = count_task_frequency(tasks_left)
    tasks_left["rel_count"] = tasks_left.abs_count / tasks_left.task_freq

    # --------------------

    data = data[data.granularity == "execution"]
    data = data[data.new_correct == False]
    data = data[[isinstance(x, str) for x in data.program]]

    data["string_square_sequence"] = square_sequences_to_strings(data.square_sequence)
    incorrect_submits = data.groupby(["task", "string_square_sequence"]).agg(
        {"program": partial(dict_of_counts, del_false=True)})

    incorrect_submits["distinct_programs"] = len_of_programs_dict(incorrect_submits.program)
    incorrect_submits["most_frequent_program"] = get_most_frequent_program(incorrect_submits.program)
    incorrect_submits["abs_count"] = count_total_abs_freq(incorrect_submits.program)
    incorrect_submits["task_freq"] = count_task_frequency(incorrect_submits)
    incorrect_submits["rel_count"] = incorrect_submits.abs_count / incorrect_submits.task_freq

    if kwargs["plot"]:
        print("incorrect submits")
        plot_frequent_wrong_programs_ratio(
            tasks=incorrect_submits[["abs_count", "rel_count", "task_freq", "most_frequent_program"]],
            abs_step=30, abs_begin=1, abs_end=11,
            rel_step=0.05, rel_begin=1, rel_end=11)
        print("leaving points")
        plot_frequent_wrong_programs_ratio(
            tasks=tasks_left[["abs_count", "rel_count", "task_freq", "most_frequent_program"]],
            abs_step=5, abs_begin=1, abs_end=11,
            rel_step=0.05, rel_begin=1, rel_end=11)

    tasks = pd.DataFrame(index=tasks_left.index.levels[0])
    tasks["frequent_leaving_points_ratio"], \
    tasks["unique_frequent_leaving_points"], \
    tasks["frequent_leaving_points_programs"] = count_frequent_wrong_programs_ratio(tasks=tasks_left,
                                                                                    abs_threshold=10,
                                                                                    rel_threshold=0.10)
    tasks["frequent_incorrect_submits_ratio"], \
    tasks["unique_frequent_incorrect_submits"], \
    tasks["frequent_incorrect_submits_programs"] = count_frequent_wrong_programs_ratio(tasks=incorrect_submits,
                                                                                       abs_threshold=50,
                                                                                       rel_threshold=0.10)

    for column in ["frequent_leaving_points_ratio", "unique_frequent_leaving_points",
                   "frequent_incorrect_submits_ratio", "unique_frequent_incorrect_submits"]:
        print(column, statistics(tasks[column]))

    return tasks[["frequent_leaving_points_ratio", "unique_frequent_leaving_points",
                  "frequent_incorrect_submits_ratio", "unique_frequent_incorrect_submits"]]


def learner_task_session_performance_measures(snapshots_path, task_sessions_path, tasks_path):
    """
    Computes task session performance measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; task session performance measures
    """
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=[])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]

    data["granularity_submits"] = data.granularity
    data["program_edits"] = data.program
    data["program_1_0"] = data.program

    ts = data.groupby("task_session").agg({"task_session": "max",
                                           "new_correct": count_true,
                                           "time_spent": lambda x: np.log(max(x)),
                                           "granularity": count_edits,
                                           "granularity_submits": count_submits,
                                           "program": partial(count_deletions, mode="all"),
                                           "program_edits": partial(count_deletions, mode="edits"),
                                           "program_1_0": partial(count_deletions, mode="1_0")})
    ts.new_correct = ts.new_correct.astype(int)

    performance = pd.DataFrame(index=ts.task_session)
    performance["time"] = ts.time_spent
    performance["edits"] = ts.granularity
    performance["submits"] = ts.granularity_submits
    performance["deletions_all"] = ts.program
    performance["deletions_edits"] = ts.program_edits
    performance["deletions_1_0"] = ts.program_1_0

    for column in performance:
        print(column, statistics(performance[column]))
    return performance


def learner_total_performance_measures(snapshots_path, task_sessions_path, tasks_path):
    """
    Computes total performance measures.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :return: pd.DataFrame; total performance measures
    """
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=["id", "level"])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]

    ts = data.groupby("task_session").agg({"task": "max",
                                           "student": "max",
                                           "level": "max",
                                           "new_correct": count_true,
                                           "program": last_with_empty_values,
                                           "time_spent": lambda x: np.log(max(x))})
    ts["new_solved"] = ts.new_correct.astype(bool)
    ts = ts[ts.new_solved]
    ts["credits"] = ts.new_solved * ts.level

    students = ts.groupby("student").agg({"task": pd.Series.nunique,
                                          "credits": "sum",
                                          "program": count_used_blocks,
                                          "time_spent": "sum"})
    students.rename(columns={"program": "used_blocks",
                             "task": "solved_tasks",
                             "time_spent": "total_time"},
                    inplace=True)

    for column in students:
        print(column, statistics(students[column]))
    return students


def measures_correlations(measures_table, method):
    """
    Computes correlation of task measures and creates its heat table.
    :param measures_table: pd.DataFrame; table of measures
    :param method: string; "pearson" or "spearman"
    :return: pd.DataFrame; correlation table of measures
    """
    correlations = measures_table.corr(method=method)
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()

    sns.clustermap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1, figsize=(8, 5))
    plt.gcf().subplots_adjust(bottom=0.35, left=0.25, right=0.75, top=0.95)
    plt.show()

    return correlations


def correlation_methods_correlations(pearson_measures_correlation, spearman_measures_correlation, full_or_triangle):
    """
    Computes correlation of correlation methods and creates its heat table.
    :param pearson_measures_correlation: pd.DataFrame; correlation table by Pearson
    :param spearman_measures_correlation: pd.DataFrame; correlation table by Spearman
    :param full_or_triangle: string; "full" or "triangle", type of correlation table processing mode
    :return: correlation table of correlation tables
    """
    if full_or_triangle == "full":
        correlations = np.corrcoef(np.ndarray.flatten(pearson_measures_correlation.as_matrix()),
                                   np.ndarray.flatten(spearman_measures_correlation.as_matrix()))
    elif full_or_triangle == "triangle":
        correlations = np.corrcoef(flattened_triangle_table_from_array(pearson_measures_correlation.as_matrix()),
                                   flattened_triangle_table_from_array(spearman_measures_correlation.as_matrix()))
    correlations = pd.DataFrame(correlations, index=["Pearson's", "Spearman's"], columns=["Pearson's", "Spearman's"])
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()

    return correlations


def all_correlations(snapshots_path, task_sessions_path, tasks_path, measures_function, **kwargs):
    """
    Computes all levels of correlation.
    :param snapshots_path: string; path to snaphots .csv file
    :param task_sessions_path: string; path to task sessions .csv file
    :param tasks_path: string; path to tasks .csv file
    :param measures_function: function; function which computes measures
    :return:
    """
    measures_table = measures_function(snapshots_path=snapshots_path,
                                       task_sessions_path=task_sessions_path,
                                       tasks_path=tasks_path,
                                       **kwargs)
    pearson_measures_correlation = measures_correlations(measures_table=measures_table,
                                                         method="pearson")
    spearman_measures_correlation = measures_correlations(measures_table=measures_table,
                                                          method="spearman")

    correlation_methods_correlations(pearson_measures_correlation=pearson_measures_correlation,
                                     spearman_measures_correlation=spearman_measures_correlation,
                                     full_or_triangle="full")

    correlation_methods_correlations(pearson_measures_correlation=pearson_measures_correlation,
                                     spearman_measures_correlation=spearman_measures_correlation,
                                     full_or_triangle="triangle")



"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=difficulty_and_complexity_measures,
                 variable_group_title="difficulty and complexity measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=solution_uniqueness_measures,
                 variable_group_title="solution uniqueness measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=task_similarity_measures,
                 variable_group_title="task similarity measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=learner_task_session_performance_measures,
                 variable_group_title="students' task performance measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=learner_total_performance_measures,
                 variable_group_title="students' total performance measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=frequent_problems_measures,
                 variable_group_title="mistakes measures",
                 plot=False)
"""



"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_qqq_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks_red_to_d.csv",
                 measures_function=frequent_problems_measures,
                 variable_group_title="mistakes measures",
                 plot=True)
"""