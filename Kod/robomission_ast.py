#!/usr/bin/env python
# -*- coding: utf-8 -*-

import betterast
import numpy as np
import zss


def add_level(tree_description, current_tree_i, current_level, levels, char):
    tree_description[levels[current_level]][0] += 1
    tree_description.append([0, char])
    current_tree_i += 1
    current_level += 1
    levels[current_level] = current_tree_i
    return tree_description, current_tree_i, current_level, levels


def process_WI_color(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    if code_string[char_i + 1] != "x":
        tree_description[levels[current_level]][0] += 1
        tree_description.append([0, "color"])
        current_tree_i += 1
        if code_string[char_i + 1] != "!":
            tree_description[levels[current_level]][0] += 1
            tree_description.append([0, "="])
            current_tree_i += 1
    return tree_description, current_tree_i, current_level, levels


def process_R(tree_description, current_tree_i, current_level, levels):
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "R")  # processing "R" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "repetitions")  # processing R "cond" level

    return tree_description, current_tree_i, current_level, levels


def process_W(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "W")  # processing "W" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "cond")  # processing W "cond" level

    tree_description, current_tree_i, current_level, levels = \
        process_WI_color(tree_description, current_tree_i, current_level, levels, code_string, char_i)  # processing of "color" and "=" nodes (not expressed directly in code)

    return tree_description, current_tree_i, current_level, levels


def process_I(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "I")  # processing "I" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "cond")  # processing I "cond" level

    tree_description, current_tree_i, current_level, levels = \
        process_WI_color(tree_description, current_tree_i, current_level, levels, code_string, char_i)  # processing of "color" and "=" nodes (not expressed directly in code)

    return tree_description, current_tree_i, current_level, levels


def process_not(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    tree_description[levels[current_level]][0] += 1  # processing "!" node
    tree_description.append([0, "!"])
    current_tree_i += 1
    if code_string[char_i + 1] == "=":  # node name for "!=" is only "!"
        skip_next = 1
    else:
        skip_next = 0
    return tree_description, current_tree_i, current_level, levels, skip_next


def process_left_bracket(tree_description, current_tree_i, current_level, levels):
    del levels[current_level]  # end "cond" level
    current_level -= 1

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "do")  # processing "do" level

    return tree_description, current_tree_i, current_level, levels


def process_right_bracket(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    if code_string[char_i + 1:char_i + 3] == "/{":  # end "do" level, processing "else" level
        del levels[current_level]
        current_level -= 1
        tree_description[levels[current_level]][0] += 1
        tree_description.append([0, "else"])
        current_tree_i += 1
        current_level += 1
        levels[current_level] = current_tree_i
        skip_next = 2
    else:
        del levels[current_level]  # end "do" level
        current_level -= 1
        del levels[current_level]  # end "R/I/W" level
        current_level -= 1
        skip_next = 0

    return tree_description, current_tree_i, current_level, levels, skip_next


def process_equals_x123456789fslrybkdg(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    tree_description[levels[current_level]][0] += 1
    if code_string[char_i] == "x":
        tree_description.append([0, "position"])
        skip_next = 0
    else:
        if code_string[char_i] in "0123456789" and len(code_string) > char_i + 1 and code_string[char_i + 1] in "0123456789":
            tree_description.append([0, str(int(code_string[char_i]) * 10 + int(code_string[char_i + 1]))])
            skip_next = 1
        else:
            tree_description.append([0, code_string[char_i]])
            skip_next = 0
    current_tree_i += 1

    return tree_description, current_tree_i, current_level, skip_next


def process_greater_less(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    if code_string[char_i + 1] == "=":  # processing ">=" node
        tree_description[levels[current_level]][0] += 1
        tree_description.append([0, "{}=".format(code_string[char_i])])
        current_tree_i += 1
        skip_next = 1
    else:
        tree_description[levels[current_level]][0] += 1  # processing ">" node
        tree_description.append([0, code_string[char_i]])
        current_tree_i += 1
        skip_next = 0

    return tree_description, current_tree_i, current_level, levels, skip_next


def build_ast(code_string, verbose=False):
    tree_description = [[0, "root"]]  # list of lists; each sublist represents one node in DFS preorder; [number of descendants, label]
    levels = {0: 0}  # dict of pointers, {num_of_level: index of parent node on selected level}
    current_level = 0  # current level of tree
    current_tree_i = 0  # index of last created node
    skip_next = 0  # number of following characters to skip

    if verbose:
        print(code_string)

    for char_i in range(0, len(code_string)):
        char = code_string[char_i]
        if skip_next:
            skip_next -= 1
            continue
        if char == "R":
            tree_description, current_tree_i, current_level, levels = \
                process_R(tree_description, current_tree_i, current_level, levels)

        elif char == "W":
            tree_description, current_tree_i, current_level, levels =\
                process_W(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char == "I":
            tree_description, current_tree_i, current_level, levels =\
                process_I(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char == "!":
            tree_description, current_tree_i, current_level, levels, skip_next = \
                process_not(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char == "{":
            tree_description, current_tree_i, current_level, levels = \
                process_left_bracket(tree_description, current_tree_i, current_level, levels)

        elif char == "}":
            tree_description, current_tree_i, current_level, levels, skip_next = \
                process_right_bracket(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char in "<>":
            tree_description, current_tree_i, current_level, levels, skip_next = \
                process_greater_less(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char in "=x123456789fslrybkdg":
            tree_description, current_tree_i, current_level, skip_next = \
                process_equals_x123456789fslrybkdg(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        else:
            raise Exception("{} not recognized as MiniRoboCode command".format(char))
        if verbose:
            print("After char: {} level: {}, tree_input: {}".format(char, current_level, tree_description))

    if current_level:
        raise Exception("Syntax error in input")

    tree = betterast.build_tree(tree_description)
    return tree

#print(build_ast("W!b{Ix<3{s}/{R5{l}}fWk{}f}", verbose=True))


def ast_ted(a_tree, b_tree):
    def binary_dist(a, b):
        if a == b:
            return 0
        return 1
    return(zss.simple_distance(a_tree, b_tree, label_dist=binary_dist))


#print(ast_ted(build_ast("W!b{rl}"), build_ast("R4{rl}")))

"""
def ast_ted_matrix_from_file(data_abs_path):
    tasks = []
    with open(data_abs_path, mode="r") as f:
        for line in f:
            if line[:3] != "id;":
                name = line.split(";")[0]
                code_string = line.split(";")[1][:-1]
                ast_builder_output = build_ast(code_string)
                tasks.append([name, code_string, ast_builder_output])
    num_tasks = range(len(tasks))
    matrix = [[0 for _ in num_tasks] for _ in num_tasks]
    for i in num_tasks:
        for j in num_tasks:
            matrix[i][j] = ast_ted(tasks[i][2], tasks[j][2])
    return tasks, matrix
"""


#my_tasks, my_matrix = ast_ted_matrix_from_file("C:/Dokumenty/Matej/MUNI/9. semestr/RoboMise/new_short-solutions-red-d.csv",)

#print(my_tasks)
#print(my_matrix)
"""
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

idx = [name for name in (my_tasks[i][1] for i in range(len(my_tasks)))]
cols = idx
df = DataFrame(my_matrix, index=idx, columns=cols)

# _r reverses the normal order of the color map 'RdYlGn'
sns.clustermap(df, figsize=(20,15))
plt.show()
#plt.savefig(fname="my_fig")
"""

