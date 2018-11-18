# INSTALOVAT PIP3 A VSECHNO PRES PIP3
#INSTALOVAT TKINTER? PRO MATPLOTLIB PYPLOT


# from collections import Counter
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
    #submits_by_tasks = all_sessions.groupby("task").agg({"granularity": "sum", "new_correct": "sum"})
    #difficulty_and_complexity["submits_incorrect"] = 1 - submits_by_tasks.new_correct / submits_by_tasks.granularity  ####################xx 1 -

    #del submits_by_tasks

    # number of block types
    distinct_blocks_by_task = all_sessions.groupby("task").agg({"solution": "last"})
    difficulty_and_complexity["distinct_blocks_1"] = count_distinct_blocks(distinct_blocks_by_task.solution, 1)
    difficulty_and_complexity.distinct_blocks_1 = difficulty_and_complexity.distinct_blocks_1.astype("int64")

    difficulty_and_complexity["distinct_blocks_3"] = count_distinct_blocks(distinct_blocks_by_task.solution, 3)
    difficulty_and_complexity.distinct_blocks_3 = difficulty_and_complexity.distinct_blocks_3.astype("int64")

    difficulty_and_complexity["distinct_blocks_4"] = count_distinct_blocks(distinct_blocks_by_task.solution, 4)
    difficulty_and_complexity.distinct_blocks_4 = difficulty_and_complexity.distinct_blocks_4.astype("int64")

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
    difficulty_and_complexity["median_submissions"] = tasks[("granularity_submits", "median")]
    difficulty_and_complexity["median_solution_length"] = tasks[("program", "median_of_lens")]
    difficulty_and_complexity["sample_solution_length"] = tasks[("solution", "len_of_last")]
    difficulty_and_complexity["deletion_ratio"] = tasks.deletion_ratio
    difficulty_and_complexity["median_deletions_all"] = tasks[("program_all", "median")]
    difficulty_and_complexity["median_deletions_line"] = tasks[("program_line", "median")]
    difficulty_and_complexity["median_deletions_bit"] = tasks[("program_bit", "median")]

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(difficulty_and_complexity)
    return difficulty_and_complexity


# Computes task solution uniqueness dataframe
# Creates task dataframe of distinct solutions, distinct visited squares sequences,
# solutions distribution entropy, visited squares sequence distribution entropy,
# sample-solution-most-frequent flag and count of program AST clusters by TED hier. clustering
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

    #tasks["sample_solution_most_frequent"] = sample_solution_not_most_frequent(tasks.solution, tasks.program)
    uniqueness = pd.DataFrame(index=tasks.index)
    uniqueness["solutions_entropy"] = list(map(entropy, tasks.program))
    uniqueness["squares_sequences_entropy"] = list(map(entropy, tasks.square_sequence))
    tasks["distinct_solutions"] = [len(x[0]) for x in tasks.program]
    tasks["distinct_squares_sequences"] = [len(x[0]) for x in tasks.square_sequence]
    uniqueness["unique_solutions"] = tasks.distinct_solutions
    uniqueness["unique_squares_sequences"] = tasks.distinct_squares_sequences
    #uniqueness["sample_solution_not_most_frequent"] = tasks.sample_solution_most_frequent  ############# is NOT most frequent!!!
    uniqueness["program_clusters_count"], _, _ = count_program_clusters(tasks.program)
    print("solutions {} {} {}".format(uniqueness.unique_solutions.quantile(0.25), uniqueness.unique_solutions.quantile(0.5), uniqueness.unique_solutions.quantile(0.75)))
    print("sequences {} {} {}".format(uniqueness.unique_squares_sequences.quantile(0.25), uniqueness.unique_squares_sequences.quantile(0.5), uniqueness.unique_squares_sequences.quantile(0.75)))
    print("clusters {} {} {}".format(uniqueness.program_clusters_count.quantile(0.25), uniqueness.program_clusters_count.quantile(0.5), uniqueness.program_clusters_count.quantile(0.75)))
    print("solutions entropy {} {} {}".format(uniqueness.solutions_entropy.quantile(0.25), uniqueness.solutions_entropy.quantile(0.5), uniqueness.solutions_entropy.quantile(0.75)))
    print("sequences entropy {} {} {}".format(uniqueness.squares_sequences_entropy.quantile(0.25), uniqueness.squares_sequences_entropy.quantile(0.5), uniqueness.squares_sequences_entropy.quantile(0.75)))
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
    similarity["closest_ast"] = get_shortest_distance(ast_ted_matrix)
    similarity["closest_levenshtein"] = get_shortest_distance(levenshtein_matrix)
    similarity["closest_blocks"] = get_shortest_distance(bag_of_blocks_matrix)
    similarity["closest_entities"] = get_shortest_distance(bag_of_entities_matrix)

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
    #performance.incorrectness = 1 - performance.incorrectness.fillna(0)  ############## INcorrectness!
    #performance["time"] = ts.time_spent
    performance["edits"] = ts.granularity
    performance["submissions"] = ts.granularity_submits
    performance["deletions_all"] = ts.program
    performance["deletions_edits"] = ts.program_line
    performance["deletions_1_0"] = ts.program_bit

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
def mistakes_measures(snapshots_path, task_sessions_path, tasks_path, **kwargs):
    data = load_extended_snapshots(snapshots_path=snapshots_path,
                                   task_sessions_path=task_sessions_path,
                                   tasks_path=tasks_path,
                                   task_sessions_cols=["id", "task"],
                                   tasks_cols=[])

    data.correct = data.correct.fillna(False)
    data.new_correct = data.new_correct.fillna(False)
    data = data[data.new_correct == data.correct]  # = snapshots whose actual correctness agree with system  # NAPSAT, ZE BYLY NESOUHLASNE VYLOUCENY

    last_ts_snapshot = data.groupby("task_session").agg({"task": "max",
                                                         "new_correct": count_true,
                                                         "granularity": "last",
                                                         "program": "last",
                                                         "square_sequence": "last"})

    last_ts_snapshot.new_correct = 0 + last_ts_snapshot.new_correct  # convert bool to int
    last_ts_snapshot["new_solved"] = last_ts_snapshot.new_correct / last_ts_snapshot.new_correct  # convert int to nan/1
    last_ts_snapshot.new_solved = last_ts_snapshot.new_solved.fillna(0)  # convert nan/1 to 0/1

    wrong_ts = last_ts_snapshot[last_ts_snapshot.new_solved == 0]
    #wrong_ts = wrong_ts.iloc[:100]    ###########
    print(wrong_ts.shape[0])

    wrong_ts = synchronous_interpreter_correctness_and_square_sequence(dataframe=wrong_ts,
                                                                       only_executions=False,
                                                                       only_edits=True,
                                                                       save=False,
                                                                       tasks_path=tasks_path)
    wrong_ts["string_square_sequence"] = square_sequences_to_strings(wrong_ts.square_sequence)

    del last_ts_snapshot
    tasks_stuck = wrong_ts.groupby(["task", "string_square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True),
                                                                            "new_solved": "count"})
    tasks_stuck["most_frequent_program"] = get_most_frequent_program(tasks_stuck.program)
    tasks_stuck["abs_count"] = count_total_abs_freq(tasks_stuck.program)
    tasks_stuck["task_freq"] = count_task_frequency(tasks_stuck)
    tasks_stuck["rel_count"] = tasks_stuck.abs_count / tasks_stuck.task_freq

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(tasks_stuck)

    # --------------------

    data = data[data.granularity == "execution"]
    data = data[data.new_correct == False]
    #data = data.iloc[:2000]  ###################
    print(data.shape[0])
    data["string_square_sequence"] = square_sequences_to_strings(data.square_sequence)
    tasks_all_wrong = data.groupby(["task", "string_square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})

    tasks_all_wrong["most_frequent_program"] = get_most_frequent_program(tasks_all_wrong.program)
    tasks_all_wrong["abs_count"] = count_total_abs_freq(tasks_all_wrong.program)
    tasks_all_wrong["task_freq"] = count_task_frequency(tasks_all_wrong)
    tasks_all_wrong["rel_count"] = tasks_all_wrong.abs_count / tasks_all_wrong.task_freq

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(tasks_all_wrong)


    if kwargs["plot"]:
        plot_frequent_wrong_programs_ratio(tasks=tasks_all_wrong[["abs_count", "rel_count", "task_freq", "most_frequent_program"]],
                                           total_sum=tasks_all_wrong.task_freq,
                                           title="All wrong submits",
                                           abs_step=30, abs_begin=1, abs_end=11,
                                           rel_step=0.05, rel_begin=1, rel_end=11)
        plot_frequent_wrong_programs_ratio(tasks=tasks_stuck[["abs_count", "rel_count", "task_freq", "most_frequent_program"]],
                                           total_sum=tasks_stuck.task_freq,
                                           title="Unsolved task sessions",
                                           abs_step=5, abs_begin=1, abs_end=11,
                                           rel_step=0.05, rel_begin=1, rel_end=11)


    tasks = pd.DataFrame(index=tasks_stuck.index.levels[0])
    tasks["stuck_frequent_programs_ratio"], tasks["stuck_unique_frequent_programs"], tasks["stuck_frequent_programs"] = count_frequent_wrong_programs_ratio(
        tasks=tasks_stuck, abs_threshold=20, rel_threshold=0.10)
    tasks["all_wrong_frequent_programs_ratio"], tasks["all_wrong_unique_frequent_programs"], tasks["all_wrong_frequent_programs"] = count_frequent_wrong_programs_ratio(
        tasks=tasks_all_wrong, abs_threshold=100, rel_threshold=0.10)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(tasks)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
    #    print(tasks_stuck[["total_wrong", "distinct_wrong", "highest_abs_count", "highest_rel_count"]])
    #    print(tasks_all_wrong[["total_wrong", "distinct_wrong", "highest_abs_count", "highest_rel_count"]])


    return tasks[["stuck_frequent_programs_ratio", "stuck_unique_frequent_programs", "all_wrong_frequent_programs_ratio", "all_wrong_unique_frequent_programs"]]

# Computes correlation of task measures and creates heat table
def measures_correlations(measures_table, method, title):
    correlations = measures_table.corr(method=method)
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    #plt.title(title)
    plt.tight_layout()
    #plt.savefig("~/dp/Obrazky/BBB.png")
    plt.show()

    sns.clustermap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1, figsize=(8,5))
    #plt.title(title)
    plt.gcf().subplots_adjust(bottom=0.35, left=0.25, right=0.75, top=0.95)
    #plt.savefig("~/dp/Obrazky/CCC.png")
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
    correlations = pd.DataFrame(correlations, index=["Pearson's", "Spearman's"], columns=["Pearson's", "Spearman's"])
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    #plt.title("""
    #Pearson's {}-matrix correlation of Pearson's and Spearman's correlation methods\n
    #applied to {}
    #""".format(full_or_triangle, variable_group_title))
    plt.tight_layout()
    #plt.gcf().subplots_adjust(left=0.3, right=0.75, top=0.8)
    plt.show()

    return correlations


# Computes correlation between matrices resulting from full and triangle mode. Does not make sense while there are only 2x2 matrices -> always correlates totally linearly.
def full_and_triangle_correlation(corr_of_full_corr_tables, corr_of_triangle_corr_tables, variable_group_title):
    correlations = np.corrcoef(np.ndarray.flatten(np.array(corr_of_full_corr_tables)),
                               np.ndarray.flatten(np.array(corr_of_triangle_corr_tables)))
    correlations = pd.DataFrame(correlations, index=["full", "triangle"], columns=["full", "triangle"])
    print(correlations)

    sns.heatmap(correlations, cmap='viridis', annot=True, vmin=-1, vmax=1)
    #plt.title("""
    #Pearson's correlation of full and triangle method\n
    #applied to {}
    #""".format(variable_group_title))
    plt.tight_layout()
    plt.show()

    return correlations


# Computes all levels of correlation
def all_correlations(snapshots_path, task_sessions_path, tasks_path, measures_function, variable_group_title, **kwargs):
    measures_table = measures_function(snapshots_path=snapshots_path,
                                       task_sessions_path=task_sessions_path,
                                       tasks_path=tasks_path,
                                       **kwargs)
    pearson_measures_correlation = measures_correlations(measures_table=measures_table,
                                                         method="pearson",
                                                         title="Pearson's correlation of {}"
                                                         .format(variable_group_title))
    spearman_measures_correlation = measures_correlations(measures_table=measures_table,
                                                          method="spearman",
                                                          title="Spearman's correlation of {}"
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


    #full_and_triangle_correlation(corr_of_full_corr_tables=corr_of_full_corr_tables,
    #                              corr_of_triangle_corr_tables=corr_of_triangle_corr_tables,
    #                              variable_group_title=variable_group_title)



"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=difficulty_measures,
                 variable_group_title="difficulty measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=complexity_measures,
                 variable_group_title="complexity measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=difficulty_and_complexity_measures,
                 variable_group_title="difficulty and complexity measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=solution_uniqueness_measures,
                 variable_group_title="solution uniqueness measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=task_similarity_measures,
                 variable_group_title="task similarity measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=student_task_performance_measures,
                 variable_group_title="students' task performance measures")
"""
"""
all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=student_total_performance_measures,
                 variable_group_title="students' total performance measures")
"""

all_correlations(snapshots_path="~/dp/Data/robomission-2018-11-03/program_snapshots_extended.csv",
                 task_sessions_path="~/dp/Data/robomission-2018-11-03/task_sessions.csv",
                 tasks_path="~/dp/Data/robomission-2018-11-03/tasks.csv",
                 measures_function=mistakes_measures,
                 variable_group_title="mistakes measures",
                 plot=True)


# TODO: VYPSAT DO DASHBOARDU SKUPINY SPRAVNYCH A SPATNYCH RESENI - REPREZENTANT = NEJCASTEJSI RESENI VE SKUPINE
# TODO: STUCK POINTS PREJMENOVAT NA LEAVING POINTS