import pandas as pd
from matplotlib import pyplot as plt

from Histograms import last, load_task_names_levels


def plot_time_operations(data, task_names_levels, save_pictures_path, num_of_tasks, time_limit, order_limit, loglog, cut_high_times, show, save):
    if cut_high_times:
        insert = ", time delta cut to 30 sec."
    else:
        insert = ""

    for task in sorted(data.task.unique()):
        print(task)
        if task <= num_of_tasks:
            task_data = data[(data.task == task)]
            task_data_cut = task_data[(task_data.time_delta_from_prev < time_limit) & (task_data.order < order_limit)]
            task_data_success = task_data[task_data.correct == True]
            task_data.to_csv("{}/csv/{} {}.csv".format(save_pictures_path, task_names_levels[task]["level"],
                                                       task_names_levels[task]["name"]))
            # print(task_data.time_spent))
            # print(task_data.order))
            ax1 = task_data.plot.scatter(x="time_delta_from_prev",
                                         y="order",
                                         c="0.7",
                                         xlim=(0, time_limit),
                                         ylim=(0, order_limit),
                                         loglog=loglog,
                                         use_index=False,
                                         title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                             insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                             len(task_data.index), len(task_data_cut.index)),
                                         grid=True,
                                         ax=plt.gca())
            ax1.set_xlabel("Total time spent")
            ax1.set_ylabel("Number of edits and submits")

            ax2 = task_data_success.plot.scatter(x="time_delta_from_prev",
                                                 y="order",
                                                 c="black",
                                                 xlim=(0, time_limit),
                                                 ylim=(0, order_limit),
                                                 loglog=loglog,
                                                 use_index=False,
                                                 title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                     insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                     len(task_data.index), len(task_data_cut.index)),
                                                 grid=True,
                                                 ax=plt.gca())
            ax2.set_xlabel("Total time spent")
            ax2.set_ylabel("Number of edits and submits")
            if show:
                plt.show()
            if save:
                plt.savefig("{}/time_vs_edits {} {}.png".format(save_pictures_path, task_names_levels[task]["level"],
                                                                task_names_levels[task]["name"]))
            plt.clf()
            print()


def time_vs_operations(snapshots_tasks_path, tasks_path, save_pictures_path, loglog, show, save, cut_high_times=False):
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    snapshots_with_tasks = pd.read_csv(snapshots_tasks_path)

    if cut_high_times:
        snapshots_with_tasks.update(pd.DataFrame({"time_delta_from_prev": [i if i <= 30 else 30 for i in snapshots_with_tasks["time_delta_from_prev"]]}))

    data = snapshots_with_tasks.groupby("task_session").agg({"task": "max",
                                                             "correct": last,
                                                             "order": "count",
                                                             "time_delta_from_prev": "sum"})
    # print(data)
    plot_time_operations(data=data, task_names_levels=task_names_levels, save_pictures_path=save_pictures_path,
                         num_of_tasks=1000, time_limit=600, order_limit=100, loglog=loglog, cut_high_times=cut_high_times,
                         show=show, save=save)


"""
time_vs_operations(snapshots_tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/joined_snapshots_tasks_delta.csv",
                   tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/tasks.csv",
                   save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Cas vs edity",
                   loglog=True,
                   cut_high_times=False,
                   show=False,
                   save=True)
"""
