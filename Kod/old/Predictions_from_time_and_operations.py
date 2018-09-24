import numpy as np
import pandas as pd
from random import choice
from Histograms import last, load_task_names_levels
from scipy.stats import percentileofscore as percentile


def level_avg_sessions(data_path, tasks_path, output_path):
    data = pd.read_csv(data_path)
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)

    data = data.groupby("task_session").agg({"task_session": "max",
                                             "task": "max",
                                             "order": "count",
                                             "time_delta_from_prev": "sum"})

    data = data.groupby("task").agg({"task_session": "count",
                                     "task": "max"})
    # print(data)

    levels = {}
    for line in data.iterrows():
        level = task_names_levels[line[1].task]["level"]
        num_ts = line[1].task_session
        # print(level)

        if level not in levels:
            levels[level] = [0, 0]
        levels[level][0] += num_ts
        levels[level][1] += 1

    with open(output_path, "w") as output:
        for level in levels:
            output.write("{}\t{}\n".format(level, levels[level][0]/levels[level][1]))


"""
level_avg_sessions(data_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/joined_snapshots_tasks_delta.csv",
                   tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/tasks.csv",
                   output_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Predikce/num_of_sessions_new_data.txt")
"""


def evaluate_percentile(number, include_0):
    if include_0:
        if number <= 1/3:
            return 1
        elif number >= 2/3:
            return 0
        else:
            return 0.5
    else:
        if number <= 0.5:
            return 0.5
        else:
            return 1


def evaluate(line, task_times):
    if line["correct"] == True:
        if line["task"] in task_times:
            if line["time_delta_from_prev"] <= np.median(task_times[line["task"]]):
                return 1
            else:
                return 0.5
        else:               # uloha zpracovavana poprve, neexistuje median
            return 1
    else:
        return 0


def predict_random(line, user_times_perc, user_ops_perc, include_0):
    if include_0:
        return choice([0, 0.5, 1])
    else:
        return choice([0.5, 1])

def predict_constant(line, user_times_perc, user_ops_perc, include_0):
    return 0.5

def predict_time_perc(line, user_times_perc, user_ops_perc, include_0):
    if line["student"] in user_times_perc:
        return evaluate_percentile(user_times_perc[line["student"]], include_0)
    else:
        return 0.5

def predict_ops_perc(line, user_times_perc, user_ops_perc, include_0):
    if line["student"] in user_ops_perc:
        return evaluate_percentile(user_ops_perc[line["student"]], include_0)
    else:
        return 0.5

def predict_medians(line, user_times_perc, user_ops_perc, include_0):
    if include_0:
        if line["student"] in user_times_perc:
            return user_times_perc[line["student"]]  # ve skutecnosti to neni time_perc, ale nejake joined_score z times i ops
        else:
            return 0.5
    else:
        if line["student"] in user_times_perc:
            if user_times_perc[line["student"]] < 0.5:  # ve skutecnosti to neni time_perc, ale nejake joined_score z times i ops
                return 0.5
            if user_times_perc[line["student"]] > 0.5:
                return 1
            else:
                return choice([0.5, 1])
        else:
            return 0.5


def basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, *args):
    if line["task"] not in task_times:
        task_times[line["task"]] = []
    task_times[line["task"]].append(line["time_delta_from_prev"])
    if line["task"] not in task_ops:
        task_ops[line["task"]] = []
    task_ops[line["task"]].append(line["order"])
    se += (pred - result) ** 2
    total += 1
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_last_time_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, *args):
    # update last time uzivatele
    if line["task"] in task_times:
        user_times_perc[line["student"]] = percentile(task_times[line["task"]], line["time_delta_from_prev"])/100
    else:
        user_times_perc[line["student"]] = 0.5
    # update median casu ulohy, update se, update total
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_last_ops_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, *args):
    # update last ops uzivatele
    if line["task"] in task_ops:
        user_ops_perc[line["student"]] = percentile(task_ops[line["task"]], line["order"])/100
    else:
        user_ops_perc[line["student"]] = 0.5
    # update median casu ulohy, update se, update total
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_ema_time_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, alpha):
    alpha = alpha
    # update time uzivatele
    if line["task"] in task_times:
        last_time_perc = percentile(task_times[line["task"]], line["time_delta_from_prev"])/100
    else:
        last_time_perc = 0.5
    if line["student"] in user_times_perc:
        user_times_perc[line["student"]] = alpha * last_time_perc + (1-alpha) * user_times_perc[line["student"]]  # exponential moving average
    else:
        user_times_perc[line["student"]] = last_time_perc
    # update median casu ulohy, update se, update total
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_ema_ops_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, alpha):
    alpha = alpha
    # update ops uzivatele
    if line["task"] in task_ops:
        last_ops_perc = percentile(task_ops[line["task"]], line["order"])/100
    else:
        last_ops_perc = 0.5
    if line["student"] in user_ops_perc:
        user_ops_perc[line["student"]] = alpha * last_ops_perc + (1-alpha) * user_ops_perc[line["student"]]  # exponential moving average
    else:
        user_ops_perc[line["student"]] = last_ops_perc
    # update median casu ulohy, update se, update total
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_last_sum_medians(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, *args):
    user_times_perc[line["student"]] = 0
    if line["task"] in task_times:
        if line["time_delta_from_prev"] <= np.median(task_times[line["task"]]):
            user_times_perc[line["student"]] += 0.5
    else:
        user_times_perc[line["student"]] += 0.5
    if line["task"] in task_ops:
        if line["order"] <= np.median(task_ops[line["task"]]):
            user_times_perc[line["student"]] += 0.5
    else:
        user_times_perc[line["student"]] += 0.5
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_ema_sum_medians(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, alpha):
    alpha = alpha
    last_score = 0
    if line["task"] in task_times:
        if line["time_delta_from_prev"] <= np.median(task_times[line["task"]]):
            last_score += 0.5
    else:
        last_score += 0.5
    if line["task"] in task_ops:
        if line["order"] <= np.median(task_ops[line["task"]]):
            last_score += 0.5
    else:
        last_score += 0.5
    if line["student"] in user_times_perc:
        user_times_perc[line["student"]] = alpha * last_score + (1-alpha) * user_times_perc[line["student"]]  # exponential moving average
    else:
        user_times_perc[line["student"]] = last_score
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_last_avg_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, *args):
    if line["task"] in task_times:
        user_times_perc[line["student"]] = (percentile(task_times[line["task"]], line["time_delta_from_prev"])/100 +
                                            percentile(task_ops[line["task"]], line["order"])/100) / 2
    else:
        user_times_perc[line["student"]] = 0.5
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total

def update_ema_avg_perc(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, alpha):
    alpha = alpha
    # update ops uzivatele
    if line["task"] in task_times:
        last_perc = (percentile(task_times[line["task"]], line["time_delta_from_prev"])/100 +
                     percentile(task_ops[line["task"]], line["order"])/100) / 2
    else:
        last_perc = 0.5
    if line["student"] in user_times_perc:
        user_times_perc[line["student"]] = alpha * last_perc + (1 - alpha) * user_times_perc[line["student"]]  # exponential moving average
    else:
        user_times_perc[line["student"]] = last_perc
    task_times, task_ops, user_times_perc, user_ops_perc, se, total = basic_update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total)
    return task_times, task_ops, user_times_perc, user_ops_perc, se, total


def main(data_path, tasks_path, variant, alpha, include_0):
    if variant == "random":
        predict = predict_random
        update = basic_update
    elif variant == "constant":
        predict = predict_constant
        update = basic_update
    elif variant == "last_time_perc":
        predict = predict_time_perc
        update = update_last_time_perc
    elif variant == "last_ops_perc":
        predict = predict_ops_perc
        update = update_last_ops_perc
    elif variant == "ema_time_perc":
        predict = predict_time_perc
        update = update_ema_time_perc
    elif variant == "ema_ops_perc":
        predict = predict_ops_perc
        update = update_ema_ops_perc
    elif variant == "last_sum_medians":
        predict = predict_medians
        update = update_last_sum_medians
    elif variant == "ema_sum_medians":
        predict = predict_medians
        update = update_ema_sum_medians
    elif variant == "last_avg_perc":
        predict = predict_time_perc
        update = update_last_avg_perc
    elif variant == "ema_avg_perc":
        predict = predict_time_perc
        update = update_ema_avg_perc
    elif variant == "NECO JINEHO JENOM DO {0.5, 1}":
        pass

    use_levels = {"moves", "world", "repeat", "while", "loops", "if"}
    data = pd.read_csv(data_path)
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)

    data = data.groupby("task_session").agg({"task_session": "max",
                                             "task": "max",
                                             "student": "max",
                                             "correct": last,
                                             "order": "count",
                                             "time_delta_from_prev": "sum"})

    for current_level in sorted(use_levels):
        task_times = {}
        task_ops = {}
        se = 0
        total = 0
        user_times_perc = {}
        user_ops_perc = {}

        for _, line in data.iterrows():
            level = task_names_levels[line["task"]]["level"]
            if level == current_level:
                pred = predict(line, user_times_perc, user_ops_perc, include_0)
                result = evaluate(line, task_times)
                task_times, task_ops, user_times_perc, user_ops_perc, se, total = update(line, pred, result, task_times, task_ops, user_times_perc, user_ops_perc, se, total, alpha)
        mse = se / total
        rmse = np.sqrt(mse)
        print("Variant:\t{}\t\talpha:\t{}\tlevel:\t{}\t\tRMSE:\t{}".format(variant, alpha, current_level, round(rmse, 5)))


for iter_alpha in range(0, 10, 1):
    main(data_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/joined_snapshots_tasks_delta.csv",
         tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-03-31/tasks.csv",
         variant="ema_avg_perc",
         alpha=iter_alpha/10,
         include_0=False)
