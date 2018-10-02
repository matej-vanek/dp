import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from Histograms import last, load_task_names_levels


def classify_by_state_frequency(data_path, tasks_path, save_pictures_path, show, save):
    data = pd.read_csv(data_path, usecols=["id", "task_session", "task", "student", "program", "granularity", "order", "correct"])
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)

    for task in range(1,87):
        print(task, task_names_levels[task]["level"], task_names_levels[task]["name"])
        task_data = data[(data.task == task)]
        task_data_edits = task_data[task_data.granularity == "edit"]

        states = task_data_edits.groupby("program").size()
        states = states.sort_values(ascending=False)
        print(states, "\n", states.sum(), "\n")
        counter = 0
        freq_states = set()
        for i in range(len(states)):
            # print(counter, freq_states)
            if counter < states.sum()/2:
                freq_states.add(states.index[i])
                counter += states.iloc[i]
            else:
                break

        output = pd.DataFrame({"frequent_ratio": [], "correct": []})
        for ts in task_data.task_session.unique():
            ts_data = task_data[task_data.task_session == ts]
            ts_data_edits = ts_data[ts_data.granularity == "edit"]
            ts_data_submits = ts_data[ts_data.granularity == "execution"]
            print(freq_states)

            frequent = 0
            total = 0
            for row in ts_data_edits.iterrows():
                print(row[1].program)
                total += 1
                if row[1].program in freq_states:
                    frequent += 1
            if total and last(ts_data_submits.correct) is not None:
                print({"frequent_ratio": frequent/total, "correct": last(ts_data_submits.correct)})
                output = output.append(pd.DataFrame({"frequent_ratio": frequent/total, "correct": last(ts_data_submits.correct)}, index=[ts]))
        # print(len(task_data.task_session.unique()))

        fig, ax = plt.subplots()
        x = (list(output[output.correct == 0].frequent_ratio), list(output[output.correct != 0].frequent_ratio))
        ax.hist(x[0], bins=10, color="red", range=(0, 1), histtype="step", label="Unsuccessfull sessions")
        ax.hist(x[1], bins=10, color="blue", range=(0, 1), histtype="step", label="Successfull sessions")
        ax.set_title("Frequent states ratio\nlevel {}, task {}\n".format(
            task_names_levels[task]["level"], task_names_levels[task]["name"]))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Frequent states ratio")
        ax.set_ylabel("Task sessions")
        plt.legend()

        if show:
            plt.show()
        if save:
            plt.savefig("{}/Frequent_states {} {}.png".format(save_pictures_path,
                                                              task_names_levels[task]["level"],
                                                              task_names_levels[task]["name"]))
        plt.clf()
        print()

"""
classify_by_state_frequency(data_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/joined_snapshots_tasks_delta.csv",
                            tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/tasks.csv",
                            save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Caste stavy 35/Caste stavy 5",
                            show=True,
                            save=False)
"""
