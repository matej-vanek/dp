import ast
from io import StringIO
import pandas as pd
import random
import re


def check_position(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                   diamonds, steps, verbose):
    """
    Checks correctness of current position, wormholes and updates diamonds info.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param pointer: int; position of processed character in program
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, diamonds; updated input variables
    """
    square_sequence.append((row_pos, col_pos))
    steps -= 1
    pointer += 1
    if (row_pos, col_pos) in diamonds:
        game_board[col_pos][row_pos] = re.sub("D", "", game_board[col_pos][row_pos])
        diamonds = list(set(diamonds) - {(row_pos, col_pos)})
    for wormhole_type in wormholes:
        if (row_pos, col_pos) in wormholes[wormhole_type]:
            row_pos, col_pos = random.choice(list(set(wormholes[wormhole_type]) - {(row_pos, col_pos)}))
            square_sequence.append((row_pos, col_pos))
    if (row_pos, col_pos) in diamonds:
        game_board[col_pos][row_pos] = re.sub("D", "", game_board[col_pos][row_pos])
        diamonds = list(set(diamonds) - {(row_pos, col_pos)})
    if is_crashed(row_pos, col_pos, game_board, verbose):
        correct = False
    return row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps


def condition_test(mode, operator, test_position, test_color, row_pos, col_pos, game_board):
    """
    Tests condition.
    :param mode: string; "position" or "color"
    :param operator: string; loaded test operator – "==", ">=", "<=", ">", "<" or "!="
    :param test_position: int; if position test, number of tested column (COUNTING FROM 1), else None
    :param test_color: string; if color test, character of tested color, else None
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :return: bool; result of test
    """
    if mode == "position":
        return eval("col_pos {} {}".format(operator, str(test_position)))
    else:
        return eval("game_board.values[row_pos][col_pos][0] {} test_color".format(operator))


def is_completed(row_pos, col_pos, game_board, diamonds, verbose):
    """
    Determines whether task is successfully completed – stands on the last line, all diamonds collected
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param verbose: bool; verbosity
    :return: bool
    """
    if "b" in game_board.values[row_pos][col_pos]:
        if len(diamonds) == 0:
            return True
        else:
            if verbose:
                print("INCORRECT - Did not collect all diamonds. Diamonds: {}".format(diamonds))
    else:
        if verbose:
            print("INCORRECT - not on the finish line, ({}, {})".format(row_pos, col_pos))
    return False


def is_crashed(row_pos, col_pos, game_board, verbose):
    """
    Determines whether rocket is out of game_board or crashed to rock.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param verbose: bool; verbosity
    :return bool;
    """
    if row_pos < 0 or row_pos > len(game_board.values) - 1:
        if verbose:
            print("CRASHED - Out of game plan rows, ({}, {})".format(row_pos, col_pos))
        return True
    if col_pos < 0 or col_pos > len(game_board.values[0]) - 1:
        if verbose:
            print("CRASHED - Out of game plan cols, ({}, {})".format(row_pos, col_pos))
        return True
    if "A" in game_board.values[row_pos][col_pos]:
        if verbose:
            print("CRASHED - Crashed to asteroid, ({}, {})".format(row_pos, col_pos))
        return True
    if "M" in game_board.values[row_pos][col_pos]:
        if verbose:
            print("CRASHED - Crashed to meteoroid, ({}, {})".format(row_pos, col_pos))
        return True
    return False


def load_game_board(tasks_path, task_id):
    """
    Loads information about initial game_board, maximal length of program and maximal shoots.
    :param tasks_path: path to tasks.csv
    :param task_id: id of task played
    :return game_board: pd.DataFrame; initial game_board state
    :return length: int; maximal length of program
    :return energy: int; maximal shoots
    """
    tasks = pd.read_csv(tasks_path)
    task = ast.literal_eval(tasks[tasks.id == task_id].setting.iloc[0])
    game_board = pd.read_csv(StringIO(re.sub("r", "d", task["fields"])),
                             names=range(0, len(task["fields"].split(";")[0].split("|"))),
                             sep="|",
                             lineterminator=";")
    # game_board indexed [cols][rows]
    # game_board.values indexed [rows][cols]
    game_board.index = range(0, len(task["fields"].split(";")))
    length = 1000
    energy = 1000
    if "length" in task:
        length = task["length"]
    if "energy" in task:
        energy = task["energy"]
    return game_board, length, energy


def load_operator_and_test_variable(pointer, program):
    """
    Loads test-containing statement information.
    :param pointer: int; position of processed character in program
    :param program: string; current program
    :return mode: string; "position", "color" or None
    :return operator: string; loaded test operator – "==", ">=", "<=", ">", "<" or "!="
    :return test_position: int; if position test, number of tested column (COUNTING FROM 1), else None
    :return test_color: string; if color test, character of tested color, else None
    :return begin_pointer: int; position of the first character of the statement body
    :return end_pointer: int; position of the last character of the statement body
    """
    operator = None
    test_position = None
    test_color = None

    if program[pointer + 1] == "x":
        mode = "position"
    elif program[pointer + 1] in "!dbgyk":
        mode = "color"
    else:
        mode = None
    if mode == "position":
        # loading operand
        if program[pointer + 3] == "=":
            begin_pointer = pointer + 4  # I+pointer, x, ?, =, begin_pointer
            if program[pointer + 2: pointer + 4] == ">=":
                operator = ">="
            elif program[pointer + 2: pointer + 4] == "<=":
                operator = "<="
            elif program[pointer + 2: pointer + 4] == "!=":
                operator = "!="
        else:
            begin_pointer = pointer + 3  # I+pointer, x, ?, begin_pointer
            if program[pointer + 2] == ">":
                operator = ">"
            elif program[pointer + 2] == "<":
                operator = "<"
            elif program[pointer + 2] == "=":
                operator = "=="
        # loading test_position
        if program[begin_pointer + 1] in "0123456789":
            test_position = 10 * int(program[begin_pointer]) + int(program[begin_pointer + 1])  # operator(s), number+begin_pointer, number, {
            test_position -= 1  # RoboMission counts from 1
            begin_pointer += 3  # operator(s), number, number, {, begin_pointer
        else:
            test_position = int(program[begin_pointer])  # operator(s), number+begin_pointer, {
            test_position -= 1  # RoboMission counts from 1
            begin_pointer += 2  # operator(s), number, {, begin_pointer
        # loading body of If statement
        if program[begin_pointer] == "}":  # operator(s), {, }+begin_pointer
            end_pointer = begin_pointer - 1
            return mode, operator, test_position, test_color, begin_pointer, end_pointer
        end_pointer = get_end_pointer(program, begin_pointer)

    elif mode == "color":
        # loading operand and test_color
        if program[pointer + 1] == "!":
            begin_pointer = pointer + 4  # I+pointer, !, color, {, begin_pointer
            test_color = program[pointer + 2]
            operator = "!="
        else:
            begin_pointer = pointer + 3  # I+pointer, color, {, begin_pointer
            test_color = program[pointer + 1]
            operator = "=="
        # loading body of If statement
        if program[begin_pointer] == "}":  # operator(s), {, }+begin_pointer
            end_pointer = begin_pointer - 1
            return mode, operator, test_position, test_color, begin_pointer, end_pointer
        end_pointer = get_end_pointer(program, begin_pointer)

    else:
        begin_pointer = pointer + 2

        if program[begin_pointer] == "}":  # operator(s), {, }+begin_pointer
            end_pointer = begin_pointer - 1
            return mode, operator, test_position, test_color, begin_pointer, end_pointer
        end_pointer = get_end_pointer(program, begin_pointer)

    return mode, operator, test_position, test_color, begin_pointer, end_pointer


def get_end_pointer(program, begin_pointer):
    """
    Computess index of the end pointer.
    :param program: string; current program
    :param begin_pointer: int; position of the first character of the statement body
    :return end_pointer: int; position of the last character of the statement body
    """
    end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer
    foreign_parentheses = 0
    while program[end_pointer] != "}" or foreign_parentheses != 0:
        if program[end_pointer] == "{":
            foreign_parentheses += 1
        elif program[end_pointer] == "}":
            foreign_parentheses -= 1
        end_pointer += 1
    end_pointer -= 1  # { begin_pointer, ..., end_pointer, }
    return end_pointer


def search_in_game_board(content, game_board, first=False):
    """
    Searches positions of game_board which contain particular character/string
    :param content: string; searched character
    :param game_board: pd.DataFrame; game_board state
    :param first: bool, if True, returns only first found result
    :return: list of 2-int-tuples OR 2-int-tuple; position(s) of found squares
    """
    results = []
    for row in range(len(game_board.values) - 1, -1, -1):
        for col in range(len(game_board.values[0])):
            if content in game_board.values[row][col]:
                if first:
                    return row, col
                results.append((row, col))
    else:
        return results


def shoot_meteoroid(row_pos, col_pos, game_board):
    """
    Updates game_board after shoot.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :return: updated game_board
    """
    diamonds_ahead = []
    for diamond in search_in_game_board("D", game_board):
        if diamond[1] == col_pos:
            if diamond[0] < row_pos:
                diamonds_ahead.append(diamond)
    for meteoroid in search_in_game_board("M", game_board):
        if meteoroid[1] == col_pos:
            if meteoroid[0] < row_pos:
                for diamond in diamonds_ahead:
                    if diamond[0] > meteoroid[0]:  # shoot to diamond
                        return game_board
                game_board[meteoroid[1]][meteoroid[0]] = re.sub("M", "", game_board[meteoroid[1]][meteoroid[0]])  # shoot to meteoroid
                return game_board
    return game_board
