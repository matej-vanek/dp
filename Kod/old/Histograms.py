import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def first(series):
    return next(series.iteritems())[1]

def last(series):
    l = None
    for item in series:
        l = item
    return l

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

def deletions(series):
    dels = 0
    last = ""
    for item in series:
        if not isinstance(item, str):
            item = ""
        if len(item) < len(last):
            dels += 1
        last = item
    return dels


def load_task_names_levels(tasks_path):
    tasks = pd.read_csv(tasks_path, usecols=["id", "name", "level"])
    task_names_levels = {task[1].id: {"name": task[1].loc["name"], "level": task[1].level} for task in tasks.iterrows()}
    return task_names_levels


def load_snapshots(snapshots_path, task_sessions_path, task_sessions_cols):
    snapshots = pd.read_csv(snapshots_path)
    task_sessions = pd.read_csv(task_sessions_path, usecols=task_sessions_cols)
    task_sessions.rename(index=str, columns={"id": "task_session"}, inplace=True)

    snapshots_with_tasks = pd.merge(snapshots, task_sessions, how="left", on="task_session")
    return snapshots_with_tasks


def histogram_from_data(data, task_names_levels, save_pictures_path, num_of_tasks, variable, limit, bins, ylim, line, title_beginning, filename_beginning, show, save):
    for task in sorted(data.task.unique()):
        print(task)
        if task <= num_of_tasks:
            task_data = data[(data.task == task)]
            task_data_cut = task_data[(data[variable] < limit)]
            task_data_success = task_data[data.correct == True]
            task_data.to_csv("{}/csv/{} {}.csv".format(save_pictures_path, task_names_levels[task]["level"],
                                                       task_names_levels[task]["name"]))
            # print(task_data)
            if line:
                task_data_unsuccess = task_data[data.correct != True]
                ax1 = task_data_unsuccess[variable].hist(bins=bins,
                                               range=(0, limit),
                                               color="red",
                                               weights=np.ones_like(task_data_unsuccess[variable]) / len(task_data[variable]),
                                               histtype="step",
                                               label="unsuccessfull")
                ax2 = task_data_success[variable].hist(bins=bins,
                                                       range=(0, limit),
                                                       color="blue",
                                                       weights=np.ones_like(task_data_success[variable]) / len(task_data[variable]),
                                                       histtype="step",
                                                       label="successful")
            else:
                ax1 = task_data[variable].hist(bins=bins,
                                               range=(0, limit),
                                               color="0.7",
                                               weights=np.ones_like(task_data[variable]) / len(task_data[variable]),
                                               label="unsuccessfull")
                ax2 = task_data_success[variable].hist(bins=bins,
                                                       range=(0, limit),
                                                       color="black",
                                                       weights=np.ones_like(task_data_success[variable]) / len(task_data[variable]),
                                                       label="successful")
            ax1.set_title("{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                title_beginning, task_names_levels[task]["level"], task_names_levels[task]["name"],
                len(task_data.index), len(task_data_cut.index)))
            ax1.set_xlim(0, limit)
            ax1.set_ylim(0, ylim)
            ax2.set_xlim(0, limit)
            ax2.set_ylim(0, ylim)
            plt.xlabel(title_beginning)
            plt.legend()

            if show:
                plt.show()
            if save:
                plt.savefig("{}/{} {} {}.png".format(save_pictures_path, filename_beginning,
                                                     task_names_levels[task]["level"], task_names_levels[task]["name"]))
            plt.clf()
            print()


def tasks_time_histograms(snapshots_tasks_path, tasks_path, save_pictures_path, show, save,
                          cut_high_times=False):
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    snapshots_with_tasks = pd.read_csv(snapshots_tasks_path)

    if cut_high_times:
        snapshots_with_tasks.update(pd.DataFrame({"time_delta_from_prev": [i if i <= 30 else 30 for i in snapshots_with_tasks["time_delta_from_prev"]]}))
        data = snapshots_with_tasks.groupby("task_session").agg({"task": "max",
                                                                 "correct": last,
                                                                 "time_delta_from_prev": "sum"})
        variable = "time_delta_from_prev"
    else:
        data = snapshots_with_tasks.groupby("task_session").agg({"task": "max",
                                                                 "correct": last,
                                                                 "time_spent": "max"})
        variable = "time_spent"
    # print(data.time_spent.quantile([0.001*i for i in range(940,951)]))

    histogram_from_data(data=data, task_names_levels=task_names_levels, save_pictures_path=save_pictures_path,
                        num_of_tasks=1000, variable=variable, limit=600, bins=60, ylim=0.4,
                        title_beginning="Total time spent", filename_beginning="time_spent",
                        show=show, save=save)


"""
tasks_time_histograms(snapshots_tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/joined_snapshots_tasks_delta.csv",
                      tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/tasks.csv",
                      save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Histogramy casu orezane",
                      show=False,
                      save=True,
                      cut_high_times=True)
"""


def tasks_num_operations_histograms(snapshots_tasks_path, tasks_path, save_pictures_path, line, show, save):
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    snapshots_with_tasks = pd.read_csv(snapshots_tasks_path)
    data = snapshots_with_tasks.groupby("task_session").agg({"task": "max",
                                                             "correct": last,
                                                             "order": "count"})
    # print(data)
    # print(data.order.quantile([0.01*i for i in range(90,101)]))

    histogram_from_data(data=data, task_names_levels=task_names_levels, save_pictures_path=save_pictures_path,
                        num_of_tasks=1000, variable="order", limit=100, bins=50, ylim=0.4, line=line,
                        title_beginning="Num of edits/submits", filename_beginning="num_edits_submits",
                        show=show, save=save)


"""
tasks_num_operations_histograms(snapshots_tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/joined_snapshots_tasks_delta.csv",
                                tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/tasks.csv",
                                save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Histogramy editu",
                                line=False,
                                show=False,
                                save=True)
"""


def tasks_operations_ratio_histograms(snapshots_tasks_path, tasks_path, save_pictures_path, line, show, save):
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    snapshots_with_tasks = pd.read_csv(snapshots_tasks_path)
    data = snapshots_with_tasks.groupby(["task_session"]).agg({"task": "max",
                                                               "correct": last,
                                                               "granularity": [count_edits, count_submits]})

    data["op_ratio"] = data[('granularity', 'count_edits')] / (data[('granularity', 'count_submits')] + data[('granularity', 'count_edits')])
    data["tasks"] = data[("task", "max")]
    data["corrects"] = data[("correct", "last")]
    print(list(data.columns.values))
    data = data[["tasks", "corrects", "op_ratio"]]
    data = data.rename(columns={"tasks": "task", "corrects": "correct"})
    print(data)
    # print(data.order.quantile([0.01*i for i in range(90,101)]))
    print(list(data.columns.values))

    histogram_from_data(data=data, task_names_levels=task_names_levels, save_pictures_path=save_pictures_path,
                        num_of_tasks=1000, variable="op_ratio", limit=1, bins=20, ylim=1, line=line,
                        title_beginning="Edits vs. total ops ratio", filename_beginning="ops_ratio",
                        show=show, save=save)

"""
tasks_operations_ratio_histograms(snapshots_tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/joined_snapshots_tasks_delta.csv",
                                tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/tasks.csv",
                                save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Histogramy/Histogramy pomeru operaci",
                                line=False,
                                show=False,
                                save=True)
"""


def tasks_deletions_histograms(snapshots_tasks_path, tasks_path, save_pictures_path, line, show, save):
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    snapshots_with_tasks = pd.read_csv(snapshots_tasks_path)
    data = snapshots_with_tasks.groupby(["task_session"]).agg({"task": "max",
                                                               "correct": last,
                                                               "program": deletions})
    # print(data)
    # print(list(data.columns.values))
    histogram_from_data(data=data, task_names_levels=task_names_levels, save_pictures_path=save_pictures_path,
                        num_of_tasks=1000, variable="program", limit=20, bins=20, ylim=1, line=line,
                        title_beginning="Deletions", filename_beginning="deletions",
                        show=show, save=save)

"""
tasks_deletions_histograms(snapshots_tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/joined_snapshots_tasks_delta.csv",
                           tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/tasks.csv",
                           save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Histogramy 345/Histogramy mazani 5 line opravene",
                           line=True,
                           show=False,
                           save=True)
"""
