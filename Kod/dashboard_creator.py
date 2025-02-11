#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matej Vanek's RoboMission dashboard creator.
Python Version: 3.6
"""

import squarify
from yattag import Doc, indent
from argparse import ArgumentParser
from collections import Counter
from matplotlib import colors, cm
from os.path import exists
from os import makedirs
from shutil import copy

from tools import *


def parse():
    """
    Parses arguments from command line.
    :return: dict; dictionary of <argument: value> pairs
    """
    parser = ArgumentParser()
    parser.add_argument('-s', '--snapshots_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission program snapshots")
    parser.add_argument('-ts', '--task_sessions_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission task sessions")
    parser.add_argument('-t', '--tasks_path', type=str, required=True,
                        help="path to .csv-like file with RoboMission tasks")
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="path to folder where outputs of dashboard will be stored")
    parser.add_argument('-r', '--replace_red', required=False, action="store_true",
                        help="replace symbol for red from 'r' to 'd' (collision with 'right')")
    parser.add_argument('-ss', '--square_sequences', required=False, action="store_true",
                        help="compute synchronous-interpreter correctness and visited square sequences for executions")

    args = vars(parser.parse_args())
    return args


def compute(args):
    """
    Computes data for the dashboard
    :param args: dict; parsed command-line arguments
    :return: tuple (tasks, correct_programs, wrong, left, learners_ts, learners_total)
    """
    if args["replace_red"]:
        print("Replacing red's 'r' symbol by 'd'")
        replace_red_by_d(file_path=args["tasks_path"],
                         output_path=args["output_path"] + "tasks_red_replaced.csv",
                         column_names=["solution"])
        args["tasks_path"] = args["output_path"] + "tasks_red_replaced.csv"

        replace_red_by_d(file_path=args["snapshots_path"],
                         output_path=args["output_path"] + "snapshots_red_replaced.csv",
                         column_names=["program"])
        args["snapshots_path"] = args["output_path"] + "snapshots_red_replaced.csv"

    if args["square_sequences"]:
        print("Running synchronous MiniRoboCode interpreter for visited squares sequences")
        synchronous_interpreter_run(
            snapshots_path=args["snapshots_path"],
            task_sessions_path=args["task_sessions_path"],
            tasks_path=args["tasks_path"],
            output_snapshots_path=args["output_path"] + "snapshots_with_sequences.csv",
        )
        args["snapshots_path"] = args["output_path"] + "snapshots_with_sequences.csv"

    if not exists(args["output_path"] + "img"):
        makedirs(args["output_path"] + "img")
    if not exists(args["output_path"] + "js"):
        makedirs(args["output_path"] + "js")
    copy("sorttable.js", args["output_path"] + "/js/sorttable.js")

    data = load_extended_snapshots(snapshots_path=args["snapshots_path"],
                                   task_sessions_path=args["task_sessions_path"],
                                   tasks_path=args["tasks_path"],
                                   task_sessions_cols=["id", "student", "solved", "time_spent", "task"],
                                   tasks_cols=["id", "name", "level", "section", "solution"])

    data.correct = data.correct.fillna(False)
    data.new_correct = data.new_correct.fillna(False)
    data = data[data.new_correct == data.correct]

    print("TASKS")
    all_sessions = data.groupby("task_session").agg({"task": "max",
                                                     "granularity": count_submits,
                                                     "new_correct": count_true,
                                                     "solution": last_with_empty_values})
    all_sessions["new_solved"] = all_sessions.new_correct.astype(int)
    all_sessions.new_solved = all_sessions.new_solved.fillna(0)
    successful_sessions = all_sessions[all_sessions.new_solved > 0]
    all_sessions = all_sessions.groupby("task").agg({"new_solved": "count"})
    successful_sessions = successful_sessions.groupby("task").agg({"new_solved": "count"})
    success_rate = successful_sessions / all_sessions
    del all_sessions
    del successful_sessions

    tasks = data[data.new_correct]
    tasks = tasks.groupby("task").agg({"name": "last",
                                       "section": "last",
                                       "task_session": "count",
                                       "solution": "last",
                                       "time_spent": "median"
                                       })
    tasks["success_rate"] = success_rate

    asts = pd.Series(list(map(build_ast, tasks.solution)), index=tasks.index)
    ast_ted_matrix = pd.DataFrame(data=None, index=sorted(tasks.index), columns=sorted(tasks.index))
    for i in ast_ted_matrix.index:
        for j in ast_ted_matrix.index:
            if i < j:
                ast_ted_matrix.loc[i][j] = ast_ted(asts.loc[i], asts.loc[j])
    flat_ast_ted_matrix = flatten_table_remove_nan(ast_ted_matrix)
    for i in ast_ted_matrix.index:
        for j in ast_ted_matrix.index:
            if i > j:
                ast_ted_matrix.loc[i][j] = ast_ted_matrix.loc[j][i]
    tasks["ast_ted_10"] = count_similar_tasks(ast_ted_matrix, np.quantile(flat_ast_ted_matrix, 0.10))
    tasks["closest_distance"], tasks["closest_task"] = get_shortest_distance(ast_ted_matrix, negative=False)
    tasks.closest_task = task_ids_to_phases(args["tasks_path"], list(tasks.closest_task))
    del flat_ast_ted_matrix
    del ast_ted_matrix

    print("CORRECT PROGRAMS")
    correct_programs = data[data.granularity == "execution"][data.new_correct]
    tasks["unique_correct_programs"] = correct_programs.groupby("task").\
        agg({"program": partial(pd.Series.nunique, dropna=True)})

    correct_programs["occurrences"] = correct_programs.program
    correct_programs.square_sequence = square_sequences_to_strings(correct_programs.square_sequence)
    correct_programs = correct_programs.groupby(["task", "square_sequence"]). \
        agg({"program": partial(dict_of_counts, del_false=True),
             "occurrences": "count"})

    correct_programs["representative"] = pd.Series([max(correct_programs.loc[i].program[0],
                                                        key=correct_programs.loc[i].program[0].get)
                                                    for i in correct_programs.index],
                                                   index=correct_programs.index)

    print("INCORRECT SUBMITS")
    wrong = data[data.granularity == "execution"]
    wrong = wrong[data.new_correct == False]
    wrong = wrong[[isinstance(x, str) for x in wrong.program]]
    wrong.square_sequence = square_sequences_to_strings(wrong.square_sequence)
    wrong = wrong.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})
    wrong["representative"] = pd.Series([max(wrong.loc[i].program[0],
                                             key=wrong.loc[i].program[0].get)
                                         for i in wrong.index],
                                        index=wrong.index)
    wrong["occurrences"] = pd.Series([sum([wrong.loc[i].program[0][solution]
                                           for solution in wrong.loc[i].program[0]])
                                      for i in wrong.index],
                                     index=wrong.index)

    print("LEAVING POINTS")
    left = data.groupby("task_session").agg({"task": "max",
                                             "new_correct": count_true,
                                             "granularity": "last",
                                             "program": last_with_empty_values,
                                             "square_sequence": last_with_empty_values})

    left = left[[isinstance(x, str) for x in left.program]]
    left["new_solved"] = left.new_correct.astype(bool)

    left = left[left.new_solved == 0]
    left = synchronous_interpreter_run(data_frame=left,
                                       only_executions=False,
                                       only_edits=True,
                                       save=False,
                                       tasks_path=args["tasks_path"])
    left.square_sequence = square_sequences_to_strings(left.square_sequence)
    left = left.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})
    left["representative"] = pd.Series([max(left.loc[i].program[0],
                                            key=left.loc[i].program[0].get)
                                        for i in left.index],
                                       index=left.index)
    left["occurrences"] = pd.Series([sum([left.loc[i].program[0][solution]
                                          for solution in left.loc[i].program[0]])
                                     for i in left.index],
                                    index=left.index)

    print("LEARNERS' TASK SESSIONS")
    data["time_log"] = data.time_spent
    learners_ts = data.groupby(["student", "task_session"]).agg({"task": "max",
                                                                 "time_spent": "max",
                                                                 "time_log": lambda x: np.log(max(x)),
                                                                 })

    print("LEARNERS' TOTAL")
    learners_total = data.groupby("task_session").agg({"task": "max",
                                                       "student": "max",
                                                       "new_correct": count_true,
                                                       "level": "max",
                                                       "solution": "last"
                                                       })
    learners_total["new_solved"] = learners_total.new_correct.astype(bool)
    learners_total = learners_total.groupby(["student", "task"]).agg({"student": "max",
                                                                      "new_solved": "max",
                                                                      "level": "max",
                                                                      "solution": "count"})
    learners_total["points"] = learners_total.new_solved * learners_total.level
    learners_total = learners_total[learners_total.new_solved]
    learners_total = learners_total.groupby("student").agg({"points": "sum",
                                                            "solution": "sum"})
    learners_total = learners_total[["points", "solution"]]

    # PLOTS
    # =====
    frequencies = [dict(Counter(tasks.ast_ted_10)), dict(Counter(tasks.closest_distance)),
                   dict(Counter(tasks.unique_correct_programs))]
    xlabels = ["AST_TED_10", "closest_distance", "unique_correct_solutions"]
    for i, variable in enumerate(frequencies):
        plt.bar(list(variable.keys()), list(variable.values()), width=0.5, color='b')
        plt.xlabel(xlabels[i])
        plt.ylabel("count")
        plt.savefig(args["output_path"] + "/img/hist_{}.png".format(xlabels[i]))
        plt.clf()

    plt.figure(figsize=(16, 10))
    squarify.plot(sizes=list(tasks.sort_values(by="section").task_session),
                  label=list(tasks.sort_values(by="section").section),
                  color=list(value_to_color(tasks.sort_values(by="section").success_rate)))
    plt.axis('off')
    plt.savefig(args["output_path"] + "/img/treemap.png")
    plt.clf()

    hists = ((tasks.time_spent, 60, 660, 30, 11, 3, 11,
              "unsuccessful", "hist_time.png", "time, 60-seconds bins", "count"),
             (correct_programs.occurrences, 5, 55, 150, 11, 15, 11,
              "occurrences", "correct_programs.png", "task's unique programs, 5-units bins", "occurrences"),
             (wrong.occurrences, 5, 55, 10000, 11, 1000, 11,
              "occurrences", "wrong_submits.png", "task's unique programs, 5-units bins", "occurrences"),
             (left.occurrences, 5, 55, 3000, 11, 300, 11,
              "occurrences", "leaving_points.png", "task's unique programs, 5-units bins", "occurrences"),
             (learners_ts.time_log, 1, 11, 20000, 11, 2000, 11,
              "time", "learners_ts_time.png", "log(time), 1-unit bins", "occurrences"),
             (learners_total.points, 10, 110, 2000, 11, 200, 11,
              "points", "learners_total_points.png", "points, 10-units bins", "occurrences"))

    for hist_variable in hists:
        plt.figure(figsize=(7, 5))
        bins = [i * hist_variable[1] for i in range(hist_variable[4])] + [1000000]
        ax = hist_variable[0].hist(bins=bins, range=(0, 1000000), color="blue", histtype="bar", label=hist_variable[7])
        ax.set_xlim(0, hist_variable[2])
        ax.set_ylim(0, hist_variable[3])
        plt.xticks([i * hist_variable[1] for i in range(hist_variable[4] + 1)])
        plt.yticks([i * hist_variable[5] for i in range(hist_variable[6])])
        plt.grid(axis="x")
        plt.xlabel(hist_variable[9])
        plt.ylabel(hist_variable[10])
        labels = [i * hist_variable[1] for i in range(hist_variable[4])]
        labels.append(u"∞")
        ax.set_xticklabels(labels)
        plt.savefig(args["output_path"] + "/img/" + hist_variable[8])
        plt.clf()
    # =====

    return tasks, correct_programs, wrong, left, learners_ts, learners_total


def value_to_color(series, yellow_highest=True):
    """
    Computes color values for cell backgrounds.
    :param series: pd.Series; series of values to transform to colors
    :param yellow_highest: bool; if True, yellow value signalizes the highest values
    :return: pd.Series; series with color hex values
    """
    value_colors = pd.Series(index=series.index)
    if yellow_highest:
        series = series.sort_values()
    else:
        series = series.sort_values(ascending=False)
    min_max = (series.iloc[len(series.index)//20],
               series.iloc[- len(series.index)//20] - series.iloc[len(series.index)//20])

    for i in value_colors.index:
        value = (series.loc[i] - min_max[0]) / float(min_max[1])
        if value > 1:
            value = 1.
        elif value < 0:
            value = 0.
        color = cm.get_cmap('viridis')(value)
        value_colors.loc[i] = colors.to_hex(color)
    return value_colors


def visualize_tasks(tasks, output_path):
    """
    Creates tasks dashboard.
    :param tasks: pd.DataFrame; tasks DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return: 
    """
    print("CREATING TASKS DASHBOARD")
    doc, tag, text = Doc().tagtext()

    success_rate_colors = value_to_color(tasks.success_rate)
    ast_ted_10_colors = value_to_color(tasks.ast_ted_10)
    closest_distance_colors = value_to_color(tasks.closest_distance, yellow_highest=False)
    unique_correct_programs_colors = value_to_color(tasks.unique_correct_programs, yellow_highest=False)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Tasks Dashboard')
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     figure {display: inline-block;}
                     figure img {vertical-align: top;}
                     figure figcaption {text-align: center;}
                     """)

        with tag('body'):
            with tag('h1'):
                text('RoboMission Task Dashboard')

            with tag('h2'):
                text('Distribution of Variables')
            with tag('p'):
                doc.stag('img', src='img/hist_time.png', alt='Histogram of median times', width="500", height="366")
                doc.stag('img', src='img/hist_AST_TED_10.png', alt='Histogram of AST TED, 10th percentile',
                         width="500", height="366")
            with tag('p'):
                doc.stag('img', src='img/hist_closest_distance.png', alt='Histogram of closest distances',
                         width="500", height="366")
                doc.stag('img', src='img/hist_unique_correct_solutions.png', alt='Histogram of unique solutions',
                         width="500", height="366")

            with tag('h2'):
                text('Tasks by the Number of Task Sessions and the Success Rate')
            with tag('p'):
                with tag('figure'):
                    doc.stag('img', src='img/treemap.png', alt="""Tasks by the number of task sessions (area) and the 
                    success rate (color, the brighter the higher)""", width="1000", height="625")
                    with tag('figcaption'):
                        text("""Tasks by number of task sessions (area) and success rate 
                        (color, the brighter the better)""")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Task', title='Write Task')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "", "filterInput1")',
                         placeholder='Name', title='Write Name')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "", "filterInput2")',
                         placeholder='Section', title='Write Section')
                doc.stag('input', type='text', id='filterInput3', onkeyup='filterTable(3, "", "filterInput3")',
                         placeholder='Sample Solution', title='Write Sample Solution')
                doc.stag('input', type='text', id='filterInput4', onkeyup='filterTable(4, "True", "filterInput4")',
                         placeholder='Total Successful Task Sessions', title='Write Total Successful Task Sessions')
                doc.stag('input', type='text', id='filterInput5', onkeyup='filterTable(5, "True", "filterInput5")',
                         placeholder='Success Rate', title='Write Success Rate')
                doc.stag('input', type='text', id='filterInput6', onkeyup='filterTable(6, "True", "filterInput6")',
                         placeholder='AST TED 10', title='Write ASt TED 10')
                doc.stag('input', type='text', id='filterInput7', onkeyup='filterTable(7, "True", "filterInput7")',
                         placeholder='Closest Distance', title='Write Closest Distance')
                doc.stag('input', type='text', id='filterInput8', onkeyup='filterTable(8, "True", "filterInput8")',
                         placeholder='Closest Task', title='Write Closest Task')
                doc.stag('input', type='text', id='filterInput9', onkeyup='filterTable(9, "True", "filterInput9")',
                         placeholder='Unique Correct Solutions', title='Write Unique Correct Solutions')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='tasksTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Task')
                        with tag('th'):
                            text('Name')
                        with tag('th'):
                            text('Section')
                        with tag('th'):
                            text('Sample Solution')
                        with tag('th'):
                            text('Total Successful Task Sessions')
                        with tag('th'):
                            text('Success Rate')
                        with tag('th'):
                            text('Number of close tasks, AST TED, 10th percentile')
                        with tag('th'):
                            text('Closest Distance')
                        with tag('th'):
                            text('Closest Task')
                        with tag('th'):
                            text('Unique Correct Solutions')
                    for i in tasks.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i)) > 1:
                                    text(i)
                                else:
                                    text('0'+str(i))
                            with tag('td'):
                                text(tasks.loc[i]['name'])
                            with tag('td'):
                                text(tasks.loc[i].section)
                            with tag('td'):
                                text(tasks.loc[i].solution)
                            with tag('td'):
                                text(int(tasks.loc[i].task_session))
                            with tag('td', bgcolor=success_rate_colors.loc[i]):
                                text(round(tasks.loc[i].success_rate, 3))
                            with tag('td', bgcolor=ast_ted_10_colors.loc[i]):
                                text(tasks.loc[i].ast_ted_10)
                            with tag('td', bgcolor=closest_distance_colors.loc[i]):
                                text(tasks.loc[i].closest_distance)
                            with tag('td'):
                                text(tasks.loc[i].closest_task)
                            with tag('td', bgcolor=unique_correct_programs_colors.loc[i]):
                                text(int(tasks.loc[i].unique_correct_programs))

            with tag('script'):
                doc.asis("""
                    function filterTable(column, float, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("tasksTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (float) {
                            cell_value = parseFloat(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass
    with open(output_path + "/tasks_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_correct_programs(correct_programs, output_path):
    """
    Creates correct programs dashboard.
    :param correct_programs: pd.DataFrame; correct programs DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return: 
    """
    print("CREATING CORRECT PROGRAMS DASHBOARD")
    doc, tag, text = Doc().tagtext()

    occurrences_colors = value_to_color(correct_programs.occurrences)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Correct Programs Dashboard')
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     """)

        with tag('body'):
            with tag('h1'):
                text('RoboMission Correct Programs Dashboard')

            with tag('h2'):
                text("Distribution of Correct Programs' Occurrences")
            with tag('p'):
                doc.stag('img', src='img/correct_programs.png', alt="Histogram of unique programs' occurrences",
                         width="500", height="366")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Task', title='Write Task')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "", "filterInput1")',
                         placeholder='Program', title='Write Program')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "True", "filterInput2")',
                         placeholder='Occurrences', title='Write Occurrences')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='correctProgramsTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Task')
                        with tag('th'):
                            text('Visited-Squares Sequence Group')
                        with tag('th'):
                            text('Occurrences')
                    for i in correct_programs.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i[0])) > 1:
                                    text(i[0])
                                else:
                                    text('0'+str(i[0]))
                            with tag('td'):
                                text(correct_programs.loc[i].representative)
                            with tag('td', bgcolor=occurrences_colors.loc[i]):
                                text(int(correct_programs.loc[i].occurrences))

            with tag('script'):
                doc.asis("""
                    function filterTable(column, integer, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("correctProgramsTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (integer) {
                            cell_value = parseInt(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass

    with open(output_path + "/correct_programs_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_wrong(wrong, output_path):
    """
    Creates incorrect submissions dashboard.
    :param wrong: pd.DataFrame; wrong submissions DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return:
    """
    print("CREATING INCORRECT SUBMITS DASHBOARD")
    doc, tag, text = Doc().tagtext()

    occurrences_colors = value_to_color(wrong.occurrences)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Incorrect Submits Dashboard')
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     """)

        with tag('body'):
            with tag('h1'):
                text('RoboMission Incorrect Submits Dashboard')

            with tag('h2'):
                text("Distribution of Incorrect Submits' Occurrences")
            with tag('p'):
                doc.stag('img', src='img/wrong_submits.png', alt="Histogram of incorrect submissions' occurrences",
                         width="500", height="366")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Task', title='Write Task')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "", "filterInput1")',
                         placeholder='Program', title='Write Program')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "True", "filterInput2")',
                         placeholder='Occurrences', title='Write Occurrences')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='wrongTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Task')
                        with tag('th'):
                            text('Visited-Squares Sequence Group')
                        with tag('th'):
                            text('Occurrences')
                    for i in wrong.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i[0])) > 1:
                                    text(i[0])
                                else:
                                    text('0' + str(i[0]))
                            with tag('td'):
                                text(wrong.loc[i].representative)
                            with tag('td', bgcolor=occurrences_colors.loc[i]):
                                text(int(wrong.loc[i].occurrences))

            with tag('script'):
                doc.asis("""
                    function filterTable(column, integer, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("wrongTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (integer) {
                            cell_value = parseInt(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass

    with open(output_path + "/incorrect_submits_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_left(left, output_path):
    """
    Creates leaving points dashboard.
    :param left: pd.DataFrame; leaving points DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return:
    """
    print("CREATING LEAVING POINTS DASHBOARD")
    doc, tag, text = Doc().tagtext()

    occurrences_colors = value_to_color(left.occurrences)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Leaving Points Dashboard')
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     """)

        with tag('body'):
            with tag('h1'):
                text('RoboMission Leaving Points Dashboard')

            with tag('h2'):
                text("Distribution of Leaving Points' Occurrences")
            with tag('p'):
                doc.stag('img', src='img/leaving_points.png', alt="Histogram of leaving points' occurrences",
                         width="500", height="366")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Task', title='Write Task')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "", "filterInput1")',
                         placeholder='Program', title='Write Program')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "True", "filterInput2")',
                         placeholder='Occurrences', title='Write Occurrences')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='leftTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Task')
                        with tag('th'):
                            text('Visited-Squares Sequence Group')
                        with tag('th'):
                            text('Occurrences')
                    for i in left.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i[0])) > 1:
                                    text(i[0])
                                else:
                                    text('0' + str(i[0]))
                            with tag('td'):
                                text(left.loc[i].representative)
                            with tag('td', bgcolor=occurrences_colors.loc[i]):
                                text(int(left.loc[i].occurrences))

            with tag('script'):
                doc.asis("""
                    function filterTable(column, integer, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("leftTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (integer) {
                            cell_value = parseInt(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass

    with open(output_path + "/leaving_points_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_learners_ts(learners_ts, output_path):
    """
    Creates learners' task sessions dashboard.
    :param learners_ts: pd.DataFrame; learners' task sessions DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return:
    """
    print("CREATING LEARNERS' TASK SESSIONS DASHBOARD")
    doc, tag, text = Doc().tagtext()

    time_spent_colors = value_to_color(learners_ts.time_spent, yellow_highest=False)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text("Learners' Task Sessions Dashboard")
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     """)

        with tag('body'):
            with tag('h1'):
                text("RoboMission Learners' Task Sessions Dashboard")

            with tag('h2'):
                text('Distribution of Time')
            with tag('p'):
                doc.stag('img', src='img/learners_ts_time.png', alt="Histogram of time",
                         width="500", height="366")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Learner', title='Write Learner')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "True", "filterInput1")',
                         placeholder='Task Session', title='Write TaskSession')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "True", "filterInput2")',
                         placeholder='Task', title='Write Task')
                doc.stag('input', type='text', id='filterInput3', onkeyup='filterTable(3, "True", "filterInput3")',
                         placeholder='Time (logarithm)', title='Write Time (logarithm)')
                doc.stag('input', type='text', id='filterInput4', onkeyup='filterTable(4, "True", "filterInput4")',
                         placeholder='Time', title='Write Time')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='learnersTSTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Learner')
                        with tag('th'):
                            text('Task Session')
                        with tag('th'):
                            text('Task ID')
                        with tag('th'):
                            text('Logarithm of time [sec]')
                        with tag('th'):
                            text('Time [sec]')
                    for i in learners_ts.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i[0])) < 6:
                                    text("0" * (6 - len(str(i[0]))) + str(i[0]))
                                else:
                                    text(i[0])
                            with tag('td'):
                                if len(str(i[1])) < 6:
                                    text("0" * (6 - len(str(i[1]))) + str(i[1]))
                                else:
                                    text(i[1])
                            with tag('td'):
                                if len(str(int(learners_ts.loc[i].task))) < 2:
                                    text("0" * (2 - len(str(learners_ts.loc[i].task))) +
                                         str(int(learners_ts.loc[i].task)))
                                else:
                                    text(learners_ts.loc[i].task)
                            with tag('td', bgcolor=time_spent_colors.loc[i]):
                                text(round(learners_ts.loc[i].time_log, 3))
                            with tag('td'):
                                text(learners_ts.loc[i].time_spent)

            with tag('script'):
                doc.asis("""
                    function filterTable(column, integer, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("learnersTSTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (integer) {
                            cell_value = parseInt(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass

    with open(output_path + "/learners_task_session_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_learners_total(learners_total, output_path):
    """
    Creates learners' total dashboard.
    :param learners_total: pd.DataFrame; learners' total DataFrame
    :param output_path: string; path to the directory where the dashboard and additional file will be stored
    :return:
    """
    print("CREATING LEARNERS' TOTAL DASHBOARD")
    doc, tag, text = Doc().tagtext()

    points_colors = value_to_color(learners_total.points)

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        doc.stag('meta', charset='utf-8')
        with tag('head'):
            with tag('title'):
                text("Learners' Total Dashboard")
            with tag('style'):
                text("""
                     table {border-spacing: 0;
                            width: 100%;
                            border: 1px solid #ddd;}
                     th, td {text-align: left;
                             padding: 16px;}
                     tr:nth-child(even) {background-color: #f2f2f2}
                     """)

        with tag('body'):
            with tag('h1'):
                text("RoboMission Learners' Total Dashboard")

            with tag('h2'):
                text('Distribution of Points')
            with tag('p'):
                doc.stag('img', src='img/learners_total_points.png', alt="Histogram of points",
                         width="500", height="366")

            with tag('h2'):
                text('Table')
            with tag('p'):
                text('Filter columns by >/</= sign and the given value; e. g. \'>35\' or \'=zig-zag\'')
                doc.stag('br')
                doc.stag('input', type='text', id='filterInput0', onkeyup='filterTable(0, "True", "filterInput0")',
                         placeholder='Learner', title='Write Learner')
                doc.stag('input', type='text', id='filterInput1', onkeyup='filterTable(1, "True", "filterInput1")',
                         placeholder='Task Sessions', title='Write Task Sessions')
                doc.stag('input', type='text', id='filterInput2', onkeyup='filterTable(2, "True", "filterInput2")',
                         placeholder='Points', title='Write Points')

            with tag('p'):
                doc.stag('br')
                text("Sort values by a click on the header")
                with tag('table', id='learnersTotalTable', klass='sortable'):
                    with tag('tr'):
                        with tag('th'):
                            text('Learner')
                        with tag('th'):
                            text('Task Sessions')
                        with tag('th'):
                            text('Points')
                    for i in learners_total.index:
                        with tag('tr'):
                            with tag('td'):
                                if len(str(i)) < 6:
                                    text("0" * (6 - len(str(i))) + str(i))
                                else:
                                    text(i)
                            with tag('td'):
                                text(int(learners_total.loc[i].solution))
                            with tag('td', bgcolor=points_colors.loc[i]):
                                text(int(learners_total.loc[i].points))
            with tag('script'):
                doc.asis("""
                    function filterTable(column, integer, input_name) {
                      var input, filter, table, tr, td, i;
                      input = document.getElementById(input_name);
                      filter = input.value;
                      table = document.getElementById("learnersTotalTable");
                      tr = table.getElementsByTagName("tr");
                      for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[column];
                        if (td && filter.length > 1) {
                          if (integer) {
                            cell_value = parseInt(td.innerHTML)
                          } else {
                            cell_value = td.innerHTML
                          }
                          console.log(cell_value)
                          console.log(filter.slice(1))
                          if (filter.charAt(0) == ">") {
                            if (cell_value > filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "=") {
                            if (cell_value == filter.slice(1)) {
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } else if (filter.charAt(0) == "<") {
                            if (cell_value < filter.slice(1)){
                              tr[i].style.display = "";
                            } else {
                              tr[i].style.display = "none";
                            }
                          } 
                        } else {
                          tr[i].style.display = "";
                        }  
                      }
                    }
                """)
            with tag('script', src='js/sorttable.js'):
                pass

    with open(output_path + "/learners_total_dashboard.html", "w") as f:
        f.write(indent(doc.getvalue()))


if __name__ == '__main__':
    args = parse()
    results = compute(args)
    visualize_tasks(results[0], args["output_path"])
    visualize_correct_programs(results[1], args["output_path"])
    visualize_wrong(results[2], args["output_path"])
    visualize_left(results[3], args["output_path"])
    visualize_learners_ts(results[4], args["output_path"])
    visualize_learners_total(results[5], args["output_path"])
