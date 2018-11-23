#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matej Vanek's RoboMission dashboard creator.
"""

"""
File name: dashboard_creator.py
Author: Matej Vanek
Created: 2018-12-
Python Version: 3.6
"""

import editdistance
import numpy as np
import pandas as pd
from yattag import Doc, indent
from argparse import ArgumentParser

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
    args = vars(parser.parse_args())
    return args



# tasks: task_id, section, sample_solution, (num_of_ts), (num_of_learners),
#        (conflict_solutions), DIFFICULTY:median_correct_time, dict_of_all_correct_solutions, SIMILARITY:levenshtein_5,
#        PROBLEMS:dict_of_leaving+dict_of_wrong_submissions
# learners: TS_PERF:time, TOTAL_PERF:solved_tasks
def compute(args):
    data = load_extended_snapshots(snapshots_path=args["snapshots_path"],
                                   task_sessions_path=args["task_sessions_path"],
                                   tasks_path=args["tasks_path"],
                                   task_sessions_cols=["id", "student", "solved", "time_spent", "task"],
                                   tasks_cols=["id", "name", "section", "solution"])

    data.correct = data.correct.fillna(False)
    data.new_correct = data.new_correct.fillna(False)
    data = data[data.new_correct == data.correct]

    tasks = data.groupby("task").agg({"name": "last",
                                      "section": "last",
                                      "solution": "last",
                                      "time_spent": "median",

                                      })

    levenshtein_matrix = pd.DataFrame(data=None, index=sorted(tasks.index), columns=sorted(tasks.index))
    for i in levenshtein_matrix.index:
        for j in levenshtein_matrix.index:
            if i < j:
                levenshtein_matrix.loc[i][j] = editdistance.eval(tasks.solution.loc[i], tasks.solution.loc[j])
    flat_levenshtein_matrix = flatten_table_remove_nan(levenshtein_matrix)
    for i in levenshtein_matrix.index:
        for j in levenshtein_matrix.index:
            if i > j:
                levenshtein_matrix.loc[i][j] = levenshtein_matrix.loc[j][i]
    tasks["levenshtein10"] = count_similar_tasks(levenshtein_matrix, np.quantile(flat_levenshtein_matrix, 0.1))
    tasks["closest_distance"], tasks["closest_task"] = get_shortest_distance(levenshtein_matrix, negative=False)


    print("CORRECT PROGRAMS")
    correct_programs = data[data.granularity == "execution"][data.new_correct]
    correct_programs.square_sequence = square_sequences_to_strings(correct_programs.square_sequence)
    correct_programs = correct_programs.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})

    correct_programs["representant"] = pd.Series([max(correct_programs.loc[i].program[0],
                                                      key=correct_programs.loc[i].program[0].get)
                                                  for i in correct_programs.index],
                                                 index=correct_programs.index)
    correct_programs["occurences"] = pd.Series([sum([correct_programs.loc[i].program[0][solution]
                                                     for solution in correct_programs.loc[i].program[0]])
                                                for i in correct_programs.index],
                                               index=correct_programs.index)

    print("WRONG")
    wrong = data[data.granularity == "execution"]
    wrong = wrong[data.new_correct == False]
    wrong.square_sequence = square_sequences_to_strings(wrong.square_sequence)
    wrong = wrong.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})
    wrong["representant"] = pd.Series([max(wrong.loc[i].program[0],
                                           key=wrong.loc[i].program[0].get)
                                       for i in wrong.index],
                                      index=wrong.index)
    wrong["occurences"] = pd.Series([sum([wrong.loc[i].program[0][solution]
                                          for solution in wrong.loc[i].program[0]])
                                     for i in wrong.index],
                                    index=wrong.index)

    print("LEFT")
    left = data.groupby("task_session").agg({"task": "max",
                                             "new_correct": count_true,
                                             "granularity": "last",
                                             "program": last_with_empty_values,
                                             "square_sequence": last_with_empty_values})

    left.new_correct = 0 + left.new_correct  # convert bool to int
    left["new_solved"] = left.new_correct / left.new_correct  # convert int to nan/1
    left.new_solved = left.new_solved.fillna(0)  # convert nan/1 to 0/1

    left = left[left.new_solved == 0]
    left = synchronous_interpreter_correctness_and_square_sequence(dataframe=left,
                                                                   only_executions=False,
                                                                   only_edits=True,
                                                                   save=False,
                                                                   tasks_path=args["tasks_path"])
    left.square_sequence = square_sequences_to_strings(left.square_sequence)
    left = left.groupby(["task", "square_sequence"]).agg({"program": partial(dict_of_counts, del_false=True)})
    left["representant"] = pd.Series([max(left.loc[i].program[0],
                                          key=left.loc[i].program[0].get)
                                      for i in left.index],
                                     index=left.index)
    left["occurences"] = pd.Series([sum([left.loc[i].program[0][solution]
                                         for solution in left.loc[i].program[0]])
                                    for i in left.index],
                                   index=left.index)


    learners_ts = data.groupby(["student", "task_session"]).agg({"task": "max",
                                                                 "time_spent": "min"})


    learners_total = data.groupby("task_session").agg({"task": "max",
                                                       "student": "max",
                                                       "new_correct": count_true})
    learners_total.new_correct = 0 + learners_total.new_correct
    learners_total["new_solved"] = learners_total.new_correct / learners_total.new_correct
    learners_total.new_solved = learners_total.new_solved.fillna(0)
    learners_total = learners_total[learners_total.new_solved > 0]
    learners_total = learners_total.groupby("student").agg({"task": pd.Series.nunique})
    learners_total = learners_total.task


    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #pd.set_option('display.width', 700)
        #print(tasks)
        #print(correct_programs)
        #print(wrong)
        #print(left)
        #print(learners_total)

    return tasks, correct_programs, wrong, left, learners_ts, learners_total


def visualize_tasks(tasks):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('RoboMission Task Dashboard')
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
                text('RoboMission Task Dashboard')
            doc.stag('input', type='text', id='filterInput', onkeyup='filterTasks()', placeholder='Task ID',
                     title='Write task id')
            with tag('p'):
                with tag('button', onclick="sortTableBy(0, 'number', 'asc')"):
                    text('Sort by task')
                with tag('button', onclick="sortTableBy(1, null, 'asc')"):
                    text('Sort by name')
                with tag('button', onclick="sortTableBy(2, null, 'asc')"):
                    text('Sort by section')
                with tag('button', onclick="sortTableBy(3, null, 'asc')"):
                    text('Sort by solution')
                with tag('button', onclick="sortTableBy(4, 'number', 'asc')"):
                    text('Sort by time median')
                with tag('button', onclick="sortTableBy(5, 'number', 'desc')"):
                    text('Sort by levenshtein10')
                with tag('button', onclick="sortTableBy(6, 'number', 'asc')"):
                    text('Sort by closest distance')
            with tag('table', id='tasksTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Task')
                    with tag('th'):
                        text('Name')
                    with tag('th'):
                        text('Section')
                    with tag('th'):
                        text('Solution')
                    with tag('th'):
                        text('Time Median')
                    with tag('th'):
                        text('Levenshtein10')
                    with tag('th'):
                        text('Closest Distance')
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
                            text(tasks.loc[i].time_spent)
                        with tag('td'):
                            text(tasks.loc[i].levenshtein10)
                        with tag('td'):
                            text(tasks.loc[i].closest_distance)

            with tag('script'):
                doc.asis("""
    function sortTableBy(column, number, order) {
      var table, rows, switching, i, x, y, shouldSwitch;
      table = document.getElementById("tasksTable");
      switching = true;
      while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[column];
          y = rows[i + 1].getElementsByTagName("TD")[column];
          if (number == 'number') {
            if (order == 'asc') {
              if (parseFloat(x.innerHTML) > parseFloat(y.innerHTML)) {
                shouldSwitch = true;
                break;
              }
            } else {
              if (parseFloat(x.innerHTML) < parseFloat(y.innerHTML)) {
                shouldSwitch = true;
                break;
              }
            }
          } else {
            if (order == 'asc') {
              if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                shouldSwitch = true;
                break;
              }
            } else {
              if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                shouldSwitch = true;
                break;
              }
            }
          }
        }
        if (shouldSwitch) {
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
        }
      }
    }
    
    function filterTasks() {
      var input, filter, table, tr, td, i;
      input = document.getElementById("filterInput");
      filter = input.value.toUpperCase();
      table = document.getElementById("tasksTable");
      tr = table.getElementsByTagName("tr");
      for (i = 0; i < tr.length; i++) {
        td = tr[i].getElementsByTagName("td")[0];
        if (td) {
          if (td.innerHTML.indexOf(filter) > -1){
            tr[i].style.display = "";
          } else {
            tr[i].style.display = "none";
          }
        }       
      }
    }
                     """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard0.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_correct_programs(correct_programs):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('RoboMission Correct Programs Dashboard')
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
            doc.stag('input', type='text', id='filterInput', onkeyup='filterCorrectPrograms()', placeholder='task_id',
                     title='Write task id')
            with tag('button', onclick="sortTableBy(0, 'number', 'asc')"):
                text('Sort by task')
            with tag('button', onclick="sortTableBy(2, 'number', 'desc')"):
                text('Sort by occurences')
            with tag('table', id='correctProgramsTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Task')
                    with tag('th'):
                        text('Program')
                    with tag('th'):
                        text('Occurences')
                for i in correct_programs.index:
                    with tag('tr'):
                        with tag('td'):
                            if len(str(i[0])) > 1:
                                text(i[0])
                            else:
                                text('0'+str(i[0]))
                        with tag('td'):
                            text(correct_programs.loc[i].representant)
                        with tag('td'):
                            text(correct_programs.loc[i].occurences)


            with tag('script'):
                doc.asis("""
        function sortTableBy(column, number, order) {
          var table, rows, switching, i, x, y, shouldSwitch;
          table = document.getElementById("correctProgramsTable");
          switching = true;
          while (switching) {
            switching = false;
            rows = table.rows;
            for (i = 1; i < (rows.length - 1); i++) {
              shouldSwitch = false;
              x = rows[i].getElementsByTagName("TD")[column];
              y = rows[i + 1].getElementsByTagName("TD")[column];
              if (number == 'number') {
                if (order == 'desc') {
                  if (parseFloat(x.innerHTML) < parseFloat(y.innerHTML)) {
                    shouldSwitch = true;
                    break;
                  }
                } else {
                  if (parseFloat(x.innerHTML) > parseFloat(y.innerHTML)) {
                    shouldSwitch = true;
                    break;
                  }
                }
              }
            }
            if (shouldSwitch) {
              rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
              switching = true;
            }
          }
        }
        function filterCorrectPrograms() {
          var input, filter, table, tr, td, i;
          input = document.getElementById("filterInput");
          filter = input.value.toUpperCase();
          table = document.getElementById("correctProgramsTable");
          tr = table.getElementsByTagName("tr");
          for (i = 0; i < tr.length; i++) {
            td = tr[i].getElementsByTagName("td")[0];
            if (td) {
              if (td.innerHTML.indexOf(filter) > -1){
                tr[i].style.display = "";
              } else {
                tr[i].style.display = "none";
              }
            }       
          }
        }
                     """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard1.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_wrong(wrong):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('RoboMission Wron Submissions Dashboard')
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
                text('RoboMission Wrong Submission Dashboard')
            doc.stag('input', type='text', id='filterInput', onkeyup='filterWrong()',
                     placeholder='task_id', title='Write task id')
            with tag('table', id='wrongTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Task')
                    with tag('th'):
                        text('Program')
                    with tag('th'):
                        text('Occurences')
                for i in wrong.index:
                    with tag('tr'):
                        with tag('td'):
                            if len(str(i[0])) > 1:
                                text(i[0])
                            else:
                                text('0' + str(i[0]))
                        with tag('td'):
                            text(wrong.loc[i].representant)
                        with tag('td'):
                            text(wrong.loc[i].occurences)

            with tag('script'):
                doc.asis("""
            function filterWrong() {
              var input, filter, table, tr, td, i;
              input = document.getElementById("filterInput");
              filter = input.value.toUpperCase();
              table = document.getElementById("wrongTable");
              tr = table.getElementsByTagName("tr");
              for (i = 0; i < tr.length; i++) {
                td = tr[i].getElementsByTagName("td")[0];
                if (td) {
                  if (td.innerHTML.indexOf(filter) > -1){
                    tr[i].style.display = "";
                  } else {
                    tr[i].style.display = "none";
                  }
                }       
              }
            }
                             """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard2.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_left(left):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('RoboMission Leaving Points Dashboard')
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
            doc.stag('input', type='text', id='filterInput', onkeyup='filterLeft()',
                     placeholder='task_id', title='Write task id')
            with tag('table', id='leftTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Task')
                    with tag('th'):
                        text('Program')
                    with tag('th'):
                        text('Occurences')
                for i in left.index:
                    with tag('tr'):
                        with tag('td'):
                            if len(str(i[0])) > 1:
                                text(i[0])
                            else:
                                text('0' + str(i[0]))
                        with tag('td'):
                            text(left.loc[i].representant)
                        with tag('td'):
                            text(left.loc[i].occurences)

            with tag('script'):
                doc.asis("""
        function filterLeft() {
          var input, filter, table, tr, td, i;
          input = document.getElementById("filterInput");
          filter = input.value.toUpperCase();
          table = document.getElementById("leftTable");
          tr = table.getElementsByTagName("tr");
          for (i = 0; i < tr.length; i++) {
            td = tr[i].getElementsByTagName("td")[0];
            if (td) {
              if (td.innerHTML.indexOf(filter) > -1){
                tr[i].style.display = "";
              } else {
                tr[i].style.display = "none";
              }
            }       
          }
        }
                         """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard3.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_learners_ts(learners_ts):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('RoboMission Learner Task Session Dashboard')
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
                text('Learners task session info')
            doc.stag('input', type='text', id='filterInput', onkeyup='filterLearnersTS()', placeholder='learner;task_session;task',
                     title='Write learner ID, task session ID and task ID')
            with tag('table', id='learnersTSTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Learner')
                    with tag('th'):
                        text('Task Session')
                    with tag('th'):
                        text('Task')
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
                            if len(str(learners_ts.loc[i].task)) < 2:
                                text("0" * (2 - len(str(learners_ts.loc[i].task))) + str(learners_ts.loc[i].task))
                            else:
                                text(learners_ts.loc[i].task)
                        with tag('td'):
                            text(learners_ts.loc[i].time_spent)


            with tag('script'):
                doc.asis("""
        function filterLearnersTS() {
          var input, filter, table, tr, td, i;
          input = document.getElementById("filterInput");
          filter = input.value.toUpperCase().split(";");
          table = document.getElementById("learnersTSTable");
          tr = table.getElementsByTagName("tr");
          for (i = 0; i < tr.length; i++) {
            td0 = tr[i].getElementsByTagName("td")[0];
            td1 = tr[i].getElementsByTagName("td")[1];
            td2 = tr[i].getElementsByTagName("td")[2];
            if (td0 && td1 && td2) {
              if (td0.innerHTML.indexOf(filter[0]) > -1 && td1.innerHTML.indexOf(filter[1]) > -1 && td2.innerHTML.indexOf(filter[2]) > -1){
                tr[i].style.display = "";
              } else {
                tr[i].style.display = "none";
              }
            }       
          }
        }
                     """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard4.html", "w") as f:
        f.write(indent(doc.getvalue()))


def visualize_learners_total(learners_total):
    doc, tag, text = Doc().tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        doc.stag('meta', charset='utf-8')
        with tag('head'):
            with tag('title'):
                text('RoboMission Learners Total Dashboard')
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
                text('Learners total info')
            doc.stag('input', type='text', id='filterInput', onkeyup='filterLearnersTotal()', placeholder='Learner\'s ID', title='Write learner\'s id')
            with tag('table', id='learnersTotalTable'):
                with tag('tr'):
                    with tag('th'):
                        text('Learner')
                    with tag('th'):
                        text('No. of Solved Tasks')
                for i in learners_total.index:
                    with tag('tr'):
                        with tag('td'):
                            if len(str(i)) < 6:
                                text("0" * (6 - len(str(i))) + str(i))
                            else:
                                text(i)
                        with tag('td'):
                            text(learners_total.loc[i])
            with tag('script'):
                doc.asis("""
    function filterLearnersTotal() {
      var input, filter, table, tr, td, i;
      input = document.getElementById("filterInput");
      filter = input.value.toUpperCase().split(";");
      table = document.getElementById("learnersTotalTable");
      tr = table.getElementsByTagName("tr");
      for (i = 0; i < tr.length; i++) {
        td = tr[i].getElementsByTagName("td")[0];
        if (td) {
          if (td.innerHTML.indexOf(filter) > -1){
            tr[i].style.display = "";
          } else {
            tr[i].style.display = "none";
          }
        }       
      }
    }
                     """)

    # print(doc.getvalue())
    with open("/home/matejvanek/dp/Prace/dashboard5.html", "w") as f:
        f.write(indent(doc.getvalue()))


if __name__ == '__main__':
    # IF RED-D AND COMPUTE SEQUENCES:...
    args = parse()
    results = compute(args)
    visualize_tasks(results[0])
    visualize_correct_programs(results[1])
    visualize_wrong(results[2])
    visualize_left(results[3])
    visualize_learners_ts(results[4])
    visualize_learners_total(results[5])

