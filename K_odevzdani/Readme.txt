Matej Vanek
Master's thesis
Appendix A - Code

All codes are developed for Python3.6 and OS Ubuntu18.04.

----------



Files:

---

dashboard_creator.py
Creates the dashboard from pre-computed data (1) or normal data (2).

Command line usage:
(1) python dashboard_creator.py -s <path/to/program_snapshots_extended.csv> -ts <path/to/task_sessions.csv> -t <path/to/tasks_red_to_d.csv> -o <path/where/to/save/dashboard/files>
or
(2) python dashboard_creator.py -s <path/to/program_snapshots.csv> -ts <path/to/task_sessions.csv> -t <path/to/tasks.csv> -o <path/where/to/save/dashboard/files/> -r -s

---

requirements.txt
Package requirements for all Python programs.

---

betterast.py
Slightly modified version of package betterast.

---

minirobocode_interpreter.py + interpreter_tools.py
MiniRoboCode synchronous interpreter.

---

robomission_ast.py
Abstract syntax trees builder and comparator.

---

correlation.py + tools.py
Anylysis of correlations.

---

sorttable.js
Imported library for sorting tables in the dashboard.
