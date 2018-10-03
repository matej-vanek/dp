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

    data = data[data["correct"] == data["new_correct"]]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": "last"})
    all_sessions["new_solved"] = all_sessions["new_correct"] / all_sessions["new_correct"]
    all_sessions["new_solved"] = all_sessions["new_solved"].fillna(0)

    # successfulness of sessions
    successful_sessions = all_sessions[all_sessions["new_solved"] > 0]

    all_sessions_by_tasks = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"new_solved": "count"})

    difficulty = successful_sessions_by_tasks / all_sessions_by_tasks
    difficulty.rename(columns={"new_solved": "task_sessions_solved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    # successfulness of submits
    submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "new_correct": "sum"})
    difficulty["submits_correct"] = submits_by_tasks["new_correct"] / submits_by_tasks["granularity"]

    del submits_by_tasks

    # number of block types
    block_types_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty["block_types"] = count_distinct_blocks(block_types_by_task["solution"], 1)
    difficulty["block_types"] = difficulty["block_types"].astype("int64")

    difficulty["block_types_flr"] = count_distinct_blocks(block_types_by_task["solution"], 3)
    difficulty["block_types_flr"] = difficulty["block_types_flr"].astype("int64")

    difficulty["block_types_flrs"] = count_distinct_blocks(block_types_by_task["solution"], 4)
    difficulty["block_types_flrs"] = difficulty["block_types_flrs"].astype("int64")

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
    data = data[data["new_correct"] == data["correct"]]  # = snapshots which actual correctness agree with system

    data["granularity_submits"] = data["granularity"]
    data["program_all"] = data["program"]
    data["program_line"] = data["program"]
    data["program_bit"] = data["program"]

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
    complexity["deletion_ratio"] = tasks["deletion_ratio"]
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

    data = data[data["correct"] == data["new_correct"]]

    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": "last"})
    all_sessions["new_solved"] = all_sessions["new_correct"] / all_sessions["new_correct"]
    all_sessions["new_solved"] = all_sessions["new_solved"].fillna(0)

    # successfulness of sessions
    successful_sessions = all_sessions[all_sessions["new_solved"] > 0]

    all_sessions_by_tasks = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions_by_tasks = successful_sessions.groupby("task").agg({"new_solved": "count"})

    difficulty_and_complexity = 1 - successful_sessions_by_tasks / all_sessions_by_tasks  ####################xx 1 -
    difficulty_and_complexity.rename(columns={"new_solved": "task_sessions_unsolved"}, inplace=True)

    del all_sessions_by_tasks
    del successful_sessions_by_tasks
    del successful_sessions

    # successfulness of submits
    submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "new_correct": "sum"})
    difficulty_and_complexity["submits_incorrect"] = 1 - submits_by_tasks["new_correct"] / submits_by_tasks["granularity"]  ####################xx 1 -

    del submits_by_tasks

    # number of block types
    block_types_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty_and_complexity["block_types"] = count_distinct_blocks(block_types_by_task["solution"], 1)
    difficulty_and_complexity["block_types"] = difficulty_and_complexity["block_types"].astype("int64")

    difficulty_and_complexity["block_types_flr"] = count_distinct_blocks(block_types_by_task["solution"], 3)
    difficulty_and_complexity["block_types_flr"] = difficulty_and_complexity["block_types_flr"].astype("int64")

    difficulty_and_complexity["block_types_flrs"] = count_distinct_blocks(block_types_by_task["solution"], 4)
    difficulty_and_complexity["block_types_flrs"] = difficulty_and_complexity["block_types_flrs"].astype("int64")

    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "student", "task", "time_spent"],
                                   tasks_cols=["id", "solution"])
    data = data.fillna(False)
    data = data[data["new_correct"] == data["correct"]]  # = snapshots which actual correctness agree with the system's one

    data["granularity_submits"] = data["granularity"]
    data["program_all"] = data["program"]
    data["program_line"] = data["program"]
    data["program_bit"] = data["program"]

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

    difficulty_and_complexity["median_time"] = tasks[("time_spent", "median")]
    difficulty_and_complexity["median_edits"] = tasks[("granularity", "median")]
    difficulty_and_complexity["median_submits"] = tasks[("granularity_submits", "median")]
    difficulty_and_complexity["median_solution_length"] = tasks[("program", "median_of_lens")]
    difficulty_and_complexity["sample_solution_length"] = tasks[("solution", "len_of_last")]
    difficulty_and_complexity["deletion_ratio"] = tasks["deletion_ratio"]
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
    data = data[data["new_correct"].notnull()]
    data = data[data["new_correct"]]
    data = data[data["new_correct"] == data["correct"]]

    tasks = data.groupby("task").agg({"program": dict_of_counts,
                                      "square_sequence": dict_of_counts,
                                      "solution": "last"})

    tasks["sample_solution_most_frequent"] = sample_solution_most_frequent(tasks["solution"], tasks["program"])
    uniqueness = pd.DataFrame(index=tasks.index)
    uniqueness["solutions_entropy"] = list(map(entropy, tasks["program"]))
    uniqueness["sequences_entropy"] = list(map(entropy, tasks["square_sequence"]))
    tasks["distinct_solutions"] = list(map(lambda x: len(x[0]), tasks["program"]))
    tasks["distinct_sequences"] = list(map(lambda x: len(x[0]), tasks["square_sequence"]))
    uniqueness["distinct_solutions"] = tasks["distinct_solutions"]
    uniqueness["distinct_sequences"] = tasks["distinct_sequences"]
    uniqueness["sample_solution_most_frequent"] = tasks["sample_solution_most_frequent"]

    uniqueness["program_clusters_count"] = count_program_clusters(tasks["program"])
    return uniqueness


def task_similarity_measures(tasks_path):
    tasks = pd.read_csv(tasks_path, index_col="id")
    sample_solutions = tasks["solution"]
    asts = pd.Series(list(map(build_ast, sample_solutions)), index=sample_solutions.index)
    bags_of_blocks = bag_of_blocks(sample_solutions)

    ast_ted_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))
    levenshtein_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))
    bag_of_blocks_matrix = pd.DataFrame(data=None, index=sorted(sample_solutions.index), columns=sorted(sample_solutions.index))

    for i in sorted(sample_solutions.index):
        print(i)
        for j in sorted(sample_solutions.index):
            if i < j:
                ast_ted_matrix.loc[i][j] = ast_ted(asts.loc[i], asts.loc[j])
                levenshtein_matrix.loc[i][j] = editdistance.eval(sample_solutions.loc[i], sample_solutions.loc[j])
                bag_of_blocks_matrix.loc[i][j] = euclidean(bags_of_blocks.loc[i], bags_of_blocks.loc[j])
            elif i == j:
                ast_ted_matrix.loc[i][j] = 0
                levenshtein_matrix.loc[i][j] = 0
                bag_of_blocks_matrix.loc[i][j] = 0
            else:

                ast_ted_matrix.loc[i][j] = ast_ted_matrix.loc[j][i]
                levenshtein_matrix.loc[i][j] = levenshtein_matrix.loc[j][i]
                bag_of_blocks_matrix.loc[i][j] = bag_of_blocks_matrix.loc[j][i]
    #print(ast_ted_matrix)

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

"""
task_similarity_measures(tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")
"""




# Computes correlation of task measures and creates heat table
def tasks_measures_correlations(task_table, method, title):
    correlations = task_table.corr(method=method)
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    sns.clustermap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    return correlations


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
all_correlations(snapshots_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                 task_sessions_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                 tasks_path="/media/matej-ubuntu/C/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                 measures_function=solution_uniqueness_measures,
                 variable_group_title="solution uniqueness measures")
"""
# TODO: KORELACNI GRAFY, napsat Tomovi o nevzorovych nejcastejsich resenich
