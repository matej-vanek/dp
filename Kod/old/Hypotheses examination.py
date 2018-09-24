from Histograms import first, last, load_task_names_levels
import pandas as pd
import numpy as np


# hypothesis: first task of each level and the last task have bigger probability of non-solving
# result:
#     - total prob. of non-solving of a task is 14 % (4), 15 % (5)
#     - prob. of non-solving of the last task is 60 % (4), 62 % (5)
#     - prob. of non-solving of the first task of each level is 22% (4), 24 % (5)
#     - prob. of non-solving any task not belonging to last two categories is 8% (4), 9 % (5)
def percentage_incorrect_last_task(data_path, tasks_path):
    data = pd.read_csv(data_path)
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    data["level"] = data.task.map(lambda x: task_names_levels[x]["level"])

    data = data.groupby("task_session").agg({"id": last,
                                             "task": last,
                                             "student": last,
                                             "correct": last,
                                             "level": last})
    #print("data:", data)

    data.replace(np.nan, False, inplace=True)
    print("ALL DATA:\n", len(data), "\n", data.correct.value_counts(), "\n")

    last_tasks = data.groupby("student").agg({"id": last,
                                              "task": last,
                                              "correct": last})
    #print("last_tasks:", last_tasks)
    print("LAST TASKS:\n", len(last_tasks), "\n", last_tasks.correct.value_counts(), "\n")

    first_of_level = data.groupby(["student","level"]).agg({"id": first,
                                                            "task": first,
                                                            "correct": first})
    #print("first_of_level:", first_of_level)
    print("FIRST TASKS OF LEVEL:\n", len(first_of_level), "\n", first_of_level.correct.value_counts(), "\n")

    #print(len(data))
    #print(len(last_tasks))
    rest = data[data.id.map(lambda x: x not in list(last_tasks.id))]
    #print("rest:", len(rest))
    #print(len(first_of_level))
    rest = rest[rest.id.map(lambda x: x not in list(first_of_level.id))]
    #print("rest:", len(rest))
    print("REST:\n", len(rest), "\n", rest.correct.value_counts(), "\n")

percentage_incorrect_last_task("C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/joined_snapshots_tasks_delta.csv",
                               "C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/tasks.csv")
