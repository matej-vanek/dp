from collections import Counter
import editdistance
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import euclidean

from Tools import *


# Computes task difficulty dataframe
# Creates task dataframe of successfulness of sessions, successfulness of submits and number of block types
def difficulty_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "solved", "time_spent"],
                                   tasks_cols=["id", "solution"])

    data = data[data.correct == data.new_correct]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": "last"})
    all_sessions["new_solved"] = all_sessions.new_correct / all_sessions.new_correct
    all_sessions.new_solved = all_sessions.new_solved.fillna(0)

    # successfulness of sessions
    successful_sessions = all_sessions[all_sessions.new_solved > 0]

    all_sessions_by_tasks = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"new_solved": "count"})

    difficulty = successful_sessions_by_tasks / all_sessions_by_tasks
    difficulty.rename(columns={"new_solved": "task_sessions_solved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    # successfulness of submits
    submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "new_correct": "sum"})
    difficulty["submits_correct"] = submits_by_tasks.new_correct / submits_by_tasks.granularity

    del submits_by_tasks

    # number of block types
    block_types_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty["block_types"] = count_distinct_blocks(block_types_by_task.solution, 1)
    difficulty.block_types = difficulty.block_types.astype("int64")

    difficulty["block_types_flr"] = count_distinct_blocks(block_types_by_task.solution, 3)
    difficulty.block_types_flr = difficulty.block_types_flr.astype("int64")

    difficulty["block_types_flrs"] = count_distinct_blocks(block_types_by_task.solution, 4)
    difficulty.block_types_flrs = difficulty.block_types_flrs.astype("int64")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(difficulty)
    return difficulty


# Computes task difficulty dataframe
# Creates task dataframe of median time, median edits, median submits, median solution length, sample solution length,
# deletion ratio, median deletions
def complexity_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=["id", "solution"])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with system

    data["granularity_submits"] = data.granularity
    data["program_all"] = data.program
    data["program_line"] = data.program
    data["program_bit"] = data.program

    task_sessions = data.groupby("task_session").agg({"task": "last",
                                                      "time_spent": "max",
                                                      "solution": "last",
                                                      "granularity": count_edits,
                                                      "granularity_submits": count_submits,
                                                      "program": "last",
                                                      "program_all": partial(count_deletions, mode="all"),
                                                      "program_line": partial(count_deletions, mode="line"),
                                                      "program_bit": partial(count_deletions, mode="bit")})

    tasks = task_sessions.groupby("task").agg({"time_spent": "median",
                                               "granularity": "median",
                                               "granularity_submits": "median",
                                               "program": median_of_lens,
                                               "solution": len_of_last,
                                               "program_all": "median",
                                               "program_line": "median",
                                               "program_bit": ["median", "sum", "count"]})
    tasks["deletion_ratio"] = tasks[("program_bit", "sum")] / tasks[("program_bit", "count")]

    complexity = pd.DataFrame()
    complexity["median_time"] = tasks[("time_spent", "median")]
    complexity["median_edits"] = tasks[("granularity", "median")]
    complexity["median_submits"] = tasks[("granularity_submits", "median")]
    complexity["median_solution_length"] = tasks[("program", "median_of_lens")]
    complexity["sample_solution_length"] = tasks[("solution", "len_of_last")]
    complexity["deletion_ratio"] = tasks.deletion_ratio
    complexity["median_deletions_all"] = tasks[("program_all", "median")]
    complexity["median_deletions_line"] = tasks[("program_line", "median")]
    complexity["median_deletions_bit"] = tasks[("program_bit", "median")]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(complexity)
    return complexity


# joined difficulty and complexity
def difficulty_and_complexity_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "solved", "time_spent"],
                                   tasks_cols=["id", "solution"])

    data = data[data.correct == data.new_correct]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": "last"})
    all_sessions["new_solved"] = all_sessions.new_correct / all_sessions.new_correct
    all_sessions.new_solved = all_sessions.new_solved.fillna(0)

    # successfulness of sessions
    successful_sessions = all_sessions[all_sessions.new_solved > 0]

    all_sessions_by_tasks = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"new_solved": "count"})

    difficulty_and_complexity = 1 - successful_sessions_by_tasks / all_sessions_by_tasks  ####################xx 1 -
    difficulty_and_complexity.rename(columns={"new_solved": "task_sessions_unsolved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    # successfulness of submits
    submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "new_correct": "sum"})
    difficulty_and_complexity["submits_incorrect"] = 1 - submits_by_tasks.new_correct / submits_by_tasks.granularity  ####################xx 1 -

    del submits_by_tasks

    # number of block types
    block_types_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty_and_complexity["block_types"] = count_distinct_blocks(block_types_by_task.solution, 1)
    difficulty_and_complexity.block_types = difficulty_and_complexity.block_types.astype("int64")

    difficulty_and_complexity["block_types_flr"] = count_distinct_blocks(block_types_by_task.solution, 3)
    difficulty_and_complexity.block_types_flr = difficulty_and_complexity.block_types_flr.astype("int64")

    difficulty_and_complexity["block_types_flrs"] = count_distinct_blocks(block_types_by_task.solution, 4)
    difficulty_and_complexity.block_types_flrs = difficulty_and_complexity.block_types_flrs.astype("int64")

    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=["id", "solution"])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with the system's one

    data["granularity_submits"] = data.granularity
    data["program_all"] = data.program
    data["program_line"] = data.program
    data["program_bit"] = data.program

    task_sessions = data.groupby("task_session").agg({"task": "last",
                                                      "time_spent": "max",
                                                      "solution": "last",
                                                      "granularity": count_edits,
                                                      "granularity_submits": count_submits,
                                                      "program": "last",
                                                      "program_all": partial(count_deletions, mode="all"),
                                                      "program_line": partial(count_deletions, mode="line"),
                                                      "program_bit": partial(count_deletions, mode="bit")})

    tasks = task_sessions.groupby("task").agg({"time_spent": "median",
                                               "granularity": "median",
                                               "granularity_submits": "median",
                                               "program": median_of_lens,
                                               "solution": len_of_last,
                                               "program_all": "median",
                                               "program_line": "median",
                                               "program_bit": ["median", "sum", "count"]})
    tasks.deletion_ratio = tasks[("program_bit", "sum")] / tasks[("program_bit", "count")]

    difficulty_and_complexity["median_time"] = tasks[("time_spent", "median")]
    difficulty_and_complexity["median_edits"] = tasks[("granularity", "median")]
    difficulty_and_complexity["median_submits"] = tasks[("granularity_submits", "median")]
    difficulty_and_complexity["median_solution_length"] = tasks[("program", "median_of_lens")]
    difficulty_and_complexity["sample_solution_length"] = tasks[("solution", "len_of_last")]
    difficulty_and_complexity["deletion_ratio"] = tasks.deletion_ratio
    difficulty_and_complexity["median_deletions_all"] = tasks[("program_all", "median")]
    difficulty_and_complexity["median_deletions_line"] = tasks[("program_line", "median")]
    difficulty_and_complexity["median_deletions_bit"] = tasks[("program_bit", "median")]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(difficulty_and_complexity)
    return difficulty_and_complexity


# Computes task solution uniqueness dataframe
# Creates task dataframe of distinct solutions, distinct visited squares sequences,
# solutions distribution entropy, visited squares sequence distribution entropy,
# sample solution most frequent flag and count of program AST clusters by TED hier. clustering
def solution_uniqueness_measures(snapshots_path, task_sessions_path, tasks_path):
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

    tasks["sample_solution_most_frequent"] = sample_solution_not_most_frequent(tasks.solution, tasks.program)
    uniqueness = pd.DataFrame(index=tasks.index)
    uniqueness["solutions_entropy"] = list(map(entropy, tasks.program))
    uniqueness["sequences_entropy"] = list(map(entropy, tasks.square_sequence))
    # tasks["distinct_solutions"] = list(map(lambda x: len(x[0]), tasks.program))
    tasks["distinct_solutions"] = [len(x[0]) for x in tasks.program]
    # tasks["distinct_sequences"] = list(map(lambda x: len(x[0]), tasks.square_sequence))
    tasks["distinct_sequences"] = [len(x[0]) for x in tasks.square_sequence]
    uniqueness["distinct_solutions"] = tasks.distinct_solutions
    uniqueness["distinct_sequences"] = tasks.distinct_sequences
    uniqueness["sample_solution_not_most_frequent"] = tasks.sample_solution_most_frequent  ############# is NOT most frequent!!!

    uniqueness["program_clusters_count"] = count_program_clusters(tasks.program)
    return uniqueness


# Computes task similarity dataframe
# Computes similarity of tasks by abstract-syntax-tree tree-edit-distance, by euclidean bag-of-used-blocks distance and
# by levensthein distance.
# Finds out how many tasks are in 1-, 5- and 10-quantile distance to the source task (w.r.t. all distances in distance matrix).
def task_similarity_measures(snapshots_path, task_sessions_path, tasks_path):
    del snapshots_path
    del task_sessions_path

    tasks = pd.read_csv(tasks_path, index_col="id")
    sample_solutions = tasks.solution
    asts = pd.Series(list(map(build_ast, sample_solutions)), index=sample_solutions.index)
    bags_of_blocks = bag_of_blocks(sample_solutions)
    bags_of_entities = bag_of_entities(tasks.setting)

    ast_ted_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))
    levenshtein_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))
    bag_of_blocks_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))
    bag_of_entities_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))

    for i in sorted(sample_solutions.index):
        print(i)
        for j in sorted(sample_solutions.index):
            if i < j:
                ast_ted_matrix.loc[i][j] = ast_ted(asts.loc[i], asts.loc[j])
                levenshtein_matrix.loc[i][j] = editdistance.eval(sample_solutions.loc[i], sample_solutions.loc[j])
                bag_of_blocks_matrix.loc[i][j] = euclidean(bags_of_blocks.loc[i], bags_of_blocks.loc[j])
                bag_of_entities_matrix.loc[i][j] = euclidean(bags_of_entities.loc[i], bags_of_entities.loc[j])

    """
    print(ast_ted_matrix)
    frequencies = dict(Counter(ast_ted_matrix.values.flatten()))
    plt.bar(list(frequencies.keys()), list(frequencies.values()), width=0.05, color='g')
    plt.title("AST TED distances distribution")
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.show()

    print(levenshtein_matrix)
    frequencies = dict(Counter(levenshtein_matrix.values.flatten()))
    plt.bar(list(frequencies.keys()), list(frequencies.values()), width=0.05, color='g')
    plt.title("Levenshtein distances distribution")
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.show()

    print(bag_of_blocks_matrix)
    frequencies = dict(Counter(bag_of_blocks_matrix.values.flatten()))
    plt.bar(list(frequencies.keys()), list(frequencies.values()), width=0.05, color='g')
    plt.title("Bag-of-blocks distances distribution")
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.show()

    print(bag_of_entities_matrix)
    frequencies = dict(Counter(bag_of_entities_matrix.values.flatten()))
    plt.bar(list(frequencies.keys()), list(frequencies.values()), width=0.05, color='g')
    plt.title("Bag-of-entities distances distribution")
    plt.xlabel("distance")
    plt.ylabel("count")
    plt.show()
    """

    flat_ast_ted_matrix = flatten_table_remove_nan(ast_ted_matrix)
    flat_levenshtein_matrix = flatten_table_remove_nan(levenshtein_matrix)
    flat_bag_of_blocks_matrix = flatten_table_remove_nan(bag_of_blocks_matrix)
    flat_bag_of_entities_matrix = flatten_table_remove_nan(bag_of_entities_matrix)

    print(flat_ast_ted_matrix)
    print(flat_levenshtein_matrix)
    print(flat_bag_of_blocks_matrix)
    print(flat_bag_of_entities_matrix)

    for i in sorted(sample_solutions.index):
        print(i)
        for j in sorted(sample_solutions.index):
            if i > j:
                ast_ted_matrix.loc[i][j] = ast_ted_matrix.loc[j][i]
                levenshtein_matrix.loc[i][j] = levenshtein_matrix.loc[j][i]
                bag_of_blocks_matrix.loc[i][j] = bag_of_blocks_matrix.loc[j][i]
                bag_of_entities_matrix.loc[i][j] = bag_of_entities_matrix.loc[j][i]

    print("AST", ast_ted_matrix)
    print("levenshtein", levenshtein_matrix)
    print("b-o-b", bag_of_blocks_matrix)
    print("b-o-e", bag_of_entities_matrix)

    similarity = pd.DataFrame(index=tasks.index)
    similarity["ast_ted1"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.01))
    similarity["ast_ted5"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.05))
    similarity["ast_ted10"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.10))
    similarity["levenshtein1"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.01))
    similarity["levenshtein5"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.05))
    similarity["levenshtein10"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.10))
    similarity["blocks1"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.01))
    similarity["blocks5"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.05))
    similarity["blocks10"] = count_similar_tasks(bag_of_blocks_matrix, np.quantile(flat_bag_of_blocks_matrix, 0.10))
    similarity["entities1"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.01))
    similarity["entities5"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.05))
    similarity["entities10"] = count_similar_tasks(bag_of_entities_matrix, np.quantile(flat_bag_of_entities_matrix, 0.10))
    similarity["shortest_ast"] = get_shortest_distance(ast_ted_matrix)
    similarity["shortest_levenshtein"] = get_shortest_distance(levenshtein_matrix)
    similarity["shortest_blocks"] = get_shortest_distance(bag_of_blocks_matrix)
    similarity["shortest_entities"] = get_shortest_distance(bag_of_entities_matrix)

    print(similarity)
    return similarity


# Computes student's task performance dataframe
# Computes task correctness, spent time, number of edits, number of submits, number of deletions
def student_task_performance_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=[])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with system

    data["granularity_submits"] = data.granularity
    data["program_line"] = data.program
    data["program_bit"] = data.program

    ts = data.groupby("task_session").agg({"task_session": "max",
                                           "new_correct": count_true,
                                           "time_spent": "max",
                                           "granularity": count_edits,
                                           "granularity_submits": count_submits,
                                           "program": partial(count_deletions, mode="all"),
                                           "program_line": partial(count_deletions, mode="line"),
                                           "program_bit": partial(count_deletions, mode="bit")})
    ts.new_correct = 0 + ts.new_correct  # transformation bool -> int

    performance = pd.DataFrame(index=ts.task_session)

    performance["incorrectness"] = ts.new_correct / ts.new_correct
    performance.incorrectness = 1 - performance.incorrectness.fillna(0)  ############## INcorrectness!
    performance["time"] = ts.time_spent
    performance["edits"] = ts.granularity
    performance["submits"] = ts.granularity_submits
    performance["deletions_all"] = ts.program
    performance["deletions_line"] = ts.program_line
    performance["deletions_bit"] = ts.program_bit

    print(performance)
    return performance


# Computes student's total performance dataframe
# Computes number of solved tasks, total credits, number of types of used blocks, total time
def student_total_performance_measures(snapshots_path, task_sessions_path, tasks_path):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=["id", "level"])
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with system

    ts = data.groupby("task_session").agg({"task": "max",
                                           "student": "max",
                                           "level": "max",
                                           "new_correct": count_true,
                                           "program": "last",
                                           "time_spent": "max"})
    ts.new_correct = 0 + ts.new_correct
    ts["new_solved"] = ts.new_correct / ts.new_correct
    ts.new_solved = ts.new_solved.fillna(0)
    ts = ts[ts.new_solved > 0]
    ts["credits"] = ts.new_solved * ts.level

    students = ts.groupby("student").agg({"task": pd.Series.nunique,
                                          "credits": "sum",
                                          "program": count_used_blocks,
                                          "time_spent": "sum"})
    students.rename(columns={"program": "used_blocks",
                             "task": "solved_tasks",
                             "time_spent": "total_time"},
                    inplace=True)

    print(students)
    return students


#
def mistakes_measures(snapshots_path, task_sessions_path, tasks_path, plot=False):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "task"],
                                   tasks_cols=[])
    data = data[data.granularity == "execution"]
    data = data.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with system

    ts_stuck = data.groupby("task_session").agg({"task": "max",
                                           "new_correct": count_true,
                                           "program": "last"})

    ts_stuck.new_correct = 0 + ts_stuck.new_correct
    ts_stuck["new_solved"] = ts_stuck.new_correct / ts_stuck.new_correct
    ts_stuck.new_solved = ts_stuck.new_solved.fillna(0)

    wrong_ts = ts_stuck[ts_stuck.new_solved == 0]
    del ts_stuck
    tasks_stuck = wrong_ts.groupby("task").agg({"program": partial(dict_of_counts, del_false=True)})

    data = data[data.new_correct == False]
    tasks_all = data.groupby("task").agg({"program": partial(dict_of_counts, del_false=True)})

    tasks_stuck, stuck_total_sum = statistics(tasks_stuck)
    tasks_all, all_total_sum = statistics(tasks_all)

    if plot:
        plot_frequent_wrong_programs_ratio(tasks=tasks_stuck, total_sum=stuck_total_sum, title="Unsolved task sessions",
                                           abs_step=2, abs_begin=1, abs_end=11,
                                           rel_step=0.02, rel_begin=1, rel_end=11)
        plot_frequent_wrong_programs_ratio(tasks=tasks_all, total_sum=all_total_sum, title="All wrong submits",
                                           abs_step=12.5, abs_begin=1, abs_end=15,
                                           rel_step=0.04, rel_begin=1, rel_end=9)

    tasks_stuck["frequent_programs_ratio"], tasks_stuck["unique_frequent_programs"] = count_frequent_wrong_programs_ratio(
        tasks=tasks_stuck, abs_threshold=5, rel_threshold=0.05)
    tasks_all["frequent_programs_ratio"], tasks_all["unique_frequent_programs"] = count_frequent_wrong_programs_ratio(
        tasks=tasks_all, abs_threshold=10, rel_threshold=0.05)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
        print(tasks_stuck[["total_wrong", "distinct_wrong", "highest_abs_count", "highest_rel_count"]])
        print(tasks_all[["total_wrong", "distinct_wrong", "highest_abs_count", "highest_rel_count"]])

    mistakes = pd.DataFrame(index=tasks_stuck.index)
    mistakes["stuck_frequent_programs_ratio"] = tasks_stuck.frequent_programs_ratio
    mistakes["stuck_unique_frequent_programs"] = tasks_stuck.unique_frequent_programs
    mistakes["all_frequent_programs_ratio"] = tasks_all.frequent_programs_ratio
    mistakes["all_unique_frequent_programs"] = tasks_all.unique_frequent_programs

    return mistakes

# Computes correlation of task measures and creates heat table
def measures_correlations(measures_table, method, title):
    correlations = measures_table.corr(method=method)
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    sns.clustermap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    return correlations


# Computes correlation of correlation methods and creates heat table
# BE AWARE of difference between cerrelation of FULL CORRELATION TABLES and of TRIANGLE CORRELATION TABLES!!!
def correlation_methods_correlations(pearson_measures_correlation, spearman_measures_correlation, variable_group_title, full_or_triangle):
    if full_or_triangle == "full":
        correlations = np.corrcoef(np.ndarray.flatten(pearson_measures_correlation.as_matrix()),
                                   np.ndarray.flatten(spearman_measures_correlation.as_matrix()))
    elif full_or_triangle == "triangle":
        correlations = np.corrcoef(flattened_triangle_table(pearson_measures_correlation.as_matrix()),
                                   flattened_triangle_table(spearman_measures_correlation.as_matrix()))
    correlations = pd.DataFrame(correlations, index=["Pearson", "Spearman"], columns=["Pearson", "Spearman"])
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title("""
    Pearson {}-matrix correlation of Pearson and Spearman correlation methods\n
    applied to {}
    """.format(full_or_triangle, variable_group_title))
    plt.show()

    return correlations


# Computes correlation between matrices resulting from full and triangle mode. Does not make sense while there are only 2x2 matrices -> always correlates totally linearly.
def full_and_triangle_correlation(corr_of_full_corr_tables, corr_of_triangle_corr_tables, variable_group_title):
    correlations = np.corrcoef(np.ndarray.flatten(np.array(corr_of_full_corr_tables)),
                               np.ndarray.flatten(np.array(corr_of_triangle_corr_tables)))
    correlations = pd.DataFrame(correlations, index=["full", "triangle"], columns=["full", "triangle"])
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title("""
    Pearson correlation of full and triangle method\n
    applied to {}
    """.format(variable_group_title))
    plt.show()

    return correlations


# Computes all levels of correlation
def all_correlations(snapshots_path, task_sessions_path, tasks_path, measures_function, variable_group_title):
    measures_table = measures_function(snapshots_path=snapshots_path,
                                       task_sessions_path=task_sessions_path,
                                       tasks_path=tasks_path)
    pearson_measures_correlation = measures_correlations(measures_table=measures_table,
                                                         method="pearson",
                                                         title="Pearson correlation of {}"
                                                         .format(variable_group_title))
    spearman_measures_correlation = measures_correlations(measures_table=measures_table,
                                                          method="spearman",
                                                          title="Spearman correlation of {}"
                                                          .format(variable_group_title))
    corr_of_full_corr_tables = \
        correlation_methods_correlations(pearson_measures_correlation=pearson_measures_correlation,
                                         spearman_measures_correlation=spearman_measures_correlation,
                                         variable_group_title=variable_group_title,
                                         full_or_triangle="full")
    corr_of_triangle_corr_tables = \
        correlation_methods_correlations(pearson_measures_correlation=pearson_measures_correlation,
                                         spearman_measures_correlation=spearman_measures_correlation,
                                         variable_group_title=variable_group_title,
                                         full_or_triangle="triangle")

    """
    full_and_triangle_correlation(corr_of_full_corr_tables=corr_of_full_corr_tables,
                                  corr_of_triangle_corr_tables=corr_of_triangle_corr_tables,
                                  variable_group_title=variable_group_title)
    """


"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=difficulty_measures,
                 variable_group_title="difficulty measures")
"""
"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=complexity_measures,
                 variable_group_title="complexity measures")
"""
"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=difficulty_and_complexity_measures,
                 variable_group_title="difficulty and complexity measures")
"""
"""
all_correlations(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=solution_uniqueness_measures,
                 variable_group_title="solution uniqueness measures")
"""
"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=task_similarity_measures,
                 variable_group_title="task similarity measures")
"""
"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=student_task_performance_measures,
                 variable_group_title="students' task performance measures")
"""
"""
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=student_total_performance_measures,
                 variable_group_title="students' total performance measures")
"""

all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=mistakes_measures,
                 variable_group_title="mistakes measures")


# TODO: KORELACNI GRAFY