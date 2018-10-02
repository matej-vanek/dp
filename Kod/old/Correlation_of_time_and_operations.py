import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns


LEVELS = ["moves", "world", "repeat", "while", "loops", "if", "comparing", "if-else", "final-challenge"]


def generate_table(item, table, cor_dict):
    print("TASK:", item)
    table[0].append(item)
    print("num of task_sessions:", cor_dict[item]["rows"])
    table[1].append(cor_dict[item]["rows"])
    print(cor_dict[item]["correlation"])
    table[2].append(cor_dict[item]["correlation"].loc["submit_count"])
    table[3].append(cor_dict[item]["correlation"].loc["edit_count"])
    table[4].append(cor_dict[item]["correlation"].loc["total_count"])
    print()
    return table


def generate_report(table, row_names, title1, title2):
    print(table)

    print("Median submit X time corr:", round(numpy.median(table[2]), 2))
    print("Mean submit X time corr:", round(numpy.mean(table[2]), 2))
    print("Median edit X time corr:", round(numpy.median(table[3]), 2))
    print("Mean edit X time corr:", round(numpy.mean(table[3]), 2))
    print("Median both X time corr:", round(numpy.median(table[4]), 2))
    print("Mean both X time corr:", round(numpy.mean(table[4]), 2))

    ax = plt.axes()
    sns.heatmap(table[2:], cmap="Spectral", annot=False, xticklabels=table[0], yticklabels=row_names, center=0, ax=ax)
    ax.set_title(title1)
    plt.show()

    print("Median number of task sessions:", round(numpy.median(table[1]), 2))
    print("Mean number of task sessions:", round(numpy.mean(table[1]), 2))

    ax = plt.axes()
    sns.heatmap(table[1:2], cmap="gray_r", annot=False, xticklabels=table[0], yticklabels=["number of task_sessions"], ax=ax)
    ax.set_title(title2)
    plt.show()


def correlation_time_edits_submits(snapshots_path, task_sessions_path, method, by_levels, tasks_path):
    def edit_count(series):
        c = 0
        for item in series:
            if item == "edit":
                c += 1
        return c

    def submit_count(series):
        c = 0
        for item in series:
            if item == "execution":
                c += 1
        return c

    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=["id", "task", "time_spent"])
    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)

    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")
    snapshots_with_tasks["granularity2"] = snapshots_with_tasks["granularity"]
    snapshots_with_tasks["granularity3"] = snapshots_with_tasks["granularity"]
    snapshots_with_tasks.rename(index=str, columns={"granularity": "edit_count",
                                                    "granularity2": "submit_count",
                                                    "granularity3": "total_count"}, inplace=True)

    data = snapshots_with_tasks.groupby("task_session").agg({"task": "max",
                                                             "edit_count": edit_count,
                                                             "submit_count": submit_count,
                                                             "total_count": "count",
                                                             "time_spent": "max"})
    print(list(data.columns.values))
    # print(data)

    by_tasks = data.groupby("task")

    cor_dict = {}
    for item in by_tasks:
        cor = item[1].corr(method=method)
        cor_dict[item[0]] = {"rows": item[1].shape[0], "correlation": cor.loc[["edit_count", "submit_count", "total_count"], "time_spent"]}

    if by_levels:
        tasks = pd.read_csv(tasks_path, usecols=["id", "name", "level"])
        tasks_by_levels = {level: [] for level in LEVELS}
        row_names = ["submits_X_spent_time", "edits_X_spent_time", "both_X_spent_time"]
        for level in LEVELS:
            tasks_by_levels[level] = [task[0] for task in tasks.iterrows() if task[1].loc["level"] == level]
        
            table = [[], [], [], [], []]
            for item in cor_dict:
                if item in tasks_by_levels[level]:
                    table = generate_table(item, table, cor_dict)
            generate_report(table, row_names, "{}, {}".format(level, method), "{}, num of task sessions".format(level))
    else:
        table = [[], [], [], [], []]
        row_names = ["submits_X_spent_time", "edits_X_spent_time", "both_X_spent_time"]
        for item in cor_dict:
            table = generate_table(item, table, cor_dict)
        generate_report(table, row_names, "all, {}".format(method), "all, num of task sessions")


"""
correlation_time_edits_submits(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/program_snapshots.csv",
                               task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/task_sessions.csv",
                               method="spearman",
                               by_levels=True,
                               tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/tasks.csv")
"""
