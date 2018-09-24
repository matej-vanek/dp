from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from random import randint
from scipy.stats import percentileofscore as percentile

from Histograms import last, load_task_names_levels


def trajectory_plot_of_simple_user(data_path, tasks_path, n_of_users, show, save, save_path):
    data = pd.read_csv(data_path)
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)

    # TODO: VYNEST DO GRAFU I CUT_HIGH_TIMES VARIANTU
    """
    if cut_high_times:
        insert = ", time delta cut to 30 sec."
        data.update(
            pd.DataFrame({"time_delta_from_prev": [i if i <= 30 else 30 for i in data["time_delta_from_prev"]]}))

    """
    # print(data)
    data = data.groupby("task_session").agg({"task": "max",
                                             "student": "max",
                                             "correct": last,
                                             "order": "count",
                                             "time_delta_from_prev": "sum"})
    i = 0
    user_dict = {}
    while i < n_of_users:
        user_id = randint(0, 10000)
        if user_id in data.student.unique():
            #print(user_id)
            user_data = data[data.student == user_id]
            if len(user_data) >= 3:
                i += 1
                #print(user_data)
                user_dict[user_id] = []
                for line in user_data.iterrows():
                    if line[1].correct is True:
                        correctness = ""
                    else:
                        correctness = "incorrect"
                    task_id = line[1].task
                    time_perc = percentile(data[data.task == line[1].task].time_delta_from_prev, line[1].time_delta_from_prev)
                    ops_perc = percentile(data[data.task == line[1].task].order, line[1].order)
                    user_dict[user_id].append(("{} {}\n{}".format(task_names_levels[task_id]["level"],
                                                                  task_names_levels[task_id]["name"],
                                                                  correctness),
                                               100 - time_perc,
                                               100 - ops_perc))
    for user_id in user_dict:
        print("{} task_sessions, student_id {}: {}".format(len(user_dict[user_id]), user_id, user_dict[user_id]))
        plt.figure(figsize=(12, 8))
        plt.title("Student's percentiles of time and operations (higher better)\nstudent {}".format(user_id))
        plt.xlabel("tasks")
        plt.ylabel("percentiles")
        plt.ylim(0,100)

        x = [i for i in range(len(user_dict[user_id]))]
        my_xticks = [record[0] for record in user_dict[user_id]]
        plt.xticks(x, my_xticks, rotation=90)

        plt.plot([user_dict[user_id][i][1] for i in range(len(user_dict[user_id]))], marker="o", label="time")
        plt.plot([user_dict[user_id][i][2] for i in range(len(user_dict[user_id]))], marker="o", label="number of operations")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig("{}/trajektorie uzivatele {}.png".format(save_path, user_id))
        plt.clf()


trajectory_plot_of_simple_user(data_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/joined_snapshots_tasks_delta.csv",
                               tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/tasks.csv",
                               n_of_users=10,
                               show=False,
                               save=True,
                               save_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Cas a edity 34/Cas a edity - trajektorie uzivatele, percentily 4")