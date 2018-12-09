#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiniRoboCode abstract syntax tree builder.
Python Version: 3.6
"""

import betterast
import zss


def add_level(tree_description, current_tree_i, current_level, levels, char):
    """
    Adds new level to tree; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param char: current program code character
    :return: tree_description, current_tree_i, current_level, levels; updated input variables
    """
    tree_description[levels[current_level]][0] += 1
    tree_description.append([0, char])
    current_tree_i += 1
    current_level += 1
    levels[current_level] = current_tree_i
    return tree_description, current_tree_i, current_level, levels


def process_color_test(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes color test; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, levels; updated input variables
    """
    if code_string[char_i + 1] != "x":
        tree_description[levels[current_level]][0] += 1
        tree_description.append([0, "color"])
        current_tree_i += 1
        if code_string[char_i + 1] != "!":
            tree_description[levels[current_level]][0] += 1
            tree_description.append([0, "="])
            current_tree_i += 1
    return tree_description, current_tree_i, current_level, levels


def process_r(tree_description, current_tree_i, current_level, levels):
    """
    Processes 'repeat' loop; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :return: tree_description, current_tree_i, current_level, levels; updated input variables
    """
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "R")  # processing "R" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "repetitions")  # processing R "cond" level

    return tree_description, current_tree_i, current_level, levels


def process_w(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes 'while' loop; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, levels; updated input variables
    """
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "W")  # processing "W" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "cond")  # processing W "cond" level

    # processing of "color" and "=" nodes (not expressed directly in code)
    tree_description, current_tree_i, current_level, levels = \
        process_color_test(tree_description, current_tree_i, current_level, levels, code_string, char_i)

    return tree_description, current_tree_i, current_level, levels


def process_i(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes 'if' condition; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, levels; updated input variables
    """
    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "I")  # processing "I" level

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "cond")  # processing I "cond" level

    # processing of "color" and "=" nodes (not expressed directly in code)
    tree_description, current_tree_i, current_level, levels = \
        process_color_test(tree_description, current_tree_i, current_level, levels, code_string, char_i)

    return tree_description, current_tree_i, current_level, levels


def process_not(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes inequality; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, levels, skip_next;
             updated input variables and number of following characters to skip
    """
    tree_description[levels[current_level]][0] += 1  # processing "!" node
    tree_description.append([0, "!"])
    current_tree_i += 1
    if code_string[char_i + 1] == "=":  # node name for "!=" is only "!"
        skip_next = 1
    else:
        skip_next = 0
    return tree_description, current_tree_i, current_level, levels, skip_next


def process_left_bracket(tree_description, current_tree_i, current_level, levels):
    """
    Processes '{' symbol; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :return: tree_description, current_tree_i, current_level, skip_next;
             updated input variables and number of following characters to skip
    """
    del levels[current_level]  # end "cond" level
    current_level -= 1

    tree_description, current_tree_i, current_level, levels = \
        add_level(tree_description, current_tree_i, current_level, levels, "do")  # processing "do" level

    return tree_description, current_tree_i, current_level, levels


def process_right_bracket(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes comparison signs; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, skip_next;
             updated input variables and number of following characters to skip
    """
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


def process_equals_x0123456789fslrybkdg(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes position comparison, numbers, colors and move commands; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, skip_next;
             updated input variables and number of following characters to skip
    """
    tree_description[levels[current_level]][0] += 1
    if code_string[char_i] == "x":
        tree_description.append([0, "position"])
        skip_next = 0
    else:
        if code_string[char_i] in "0123456789" and \
                len(code_string) > char_i + 1 and \
                code_string[char_i + 1] in "0123456789":
            tree_description.append([0, str(int(code_string[char_i]) * 10 + int(code_string[char_i + 1]))])
            skip_next = 1
        else:
            tree_description.append([0, code_string[char_i]])
            skip_next = 0
    current_tree_i += 1

    return tree_description, current_tree_i, current_level, skip_next


def process_greater_less(tree_description, current_tree_i, current_level, levels, code_string, char_i):
    """
    Processes comparison signs; AST builder helper function.
    :param tree_description: list of nodes (node = [int, string]); DFS preorder description of tree
           [index_of_parent_node, name]
    :param current_tree_i: int; index of last created node
    :param current_level: int; current level of tree
    :param levels: dict {int: int}; {level: index_of_parent_node}
    :param code_string: string; MiniRoboCode program
    :param char_i: index of current string character
    :return: tree_description, current_tree_i, current_level, levels, skip_next;
             updated input variables and number of following characters to skip
    """
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
    """
    Builds abstract syntax tree of RoboMission MiniRoboCode program.
    :param code_string: string; MiniRoboCode program
    :param verbose: bool; verbosity
    :return: abstract syntax tree of program
    """
    tree_description = [[0, "root"]]
    levels = {0: 0}
    current_level = 0
    current_tree_i = 0
    skip_next = 0

    if verbose:
        print(code_string)

    for char_i in range(0, len(code_string)):
        char = code_string[char_i]
        if skip_next:
            skip_next -= 1
            continue
        if char == "R":
            tree_description, current_tree_i, current_level, levels = \
                process_r(tree_description, current_tree_i, current_level, levels)

        elif char == "W":
            tree_description, current_tree_i, current_level, levels =\
                process_w(tree_description, current_tree_i, current_level, levels, code_string, char_i)

        elif char == "I":
            tree_description, current_tree_i, current_level, levels =\
                process_i(tree_description, current_tree_i, current_level, levels, code_string, char_i)

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
                process_equals_x0123456789fslrybkdg(tree_description, current_tree_i, current_level, levels,
                                                    code_string, char_i)
        else:
            raise Exception("{} not recognized as MiniRoboCode command".format(char))
        if verbose:
            print("After char: {} level: {}, tree_input: {}".format(char, current_level, tree_description))

    if current_level:
        raise Exception("Syntax error in input")

    tree = betterast.build_tree(tree_description)
    return tree


def ast_ted(a_tree, b_tree):
    """
    Computes tree edit distance of MiniRoboCode abstract syntax trees
    :param a_tree: MinoRoboCode AST; first tree
    :param b_tree: MinoRoboCode AST; second tree
    :return: int; tree edit distance
    """
    def binary_dist(a, b):
        if a == b:
            return 0
        return 1
    return zss.simple_distance(a_tree, b_tree, label_dist=binary_dist)
