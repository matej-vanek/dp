import pandas as pd


def delta_from_previous(data):
    last_times = {}
    time_diff = []
    for row in data.iterrows():
        ts = row[1].task_session
        time = row[1].time_from_start
        if ts not in last_times:
            last_times[ts] = 0
        time_diff.append(time - last_times[ts])
        last_times[ts] = time
    return pd.Series(time_diff)




"""
from Histograms import load_snapshots
my_data = load_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/program_snapshots.csv",
                         task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/task_sessions.csv",
                         task_sessions_cols=["id", "task", "student", "time_spent"])
my_data["time_delta_from_prev"] = delta_from_previous(my_data)
my_data.to_csv("C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-04-28/joined_snapshots_tasks_delta.csv")
"""
