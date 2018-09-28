import ast
from io import StringIO
import pandas as pd
import re


def load_game_plan(tasks_path, task_id):
    tasks = pd.read_csv(tasks_path)
    task = ast.literal_eval(tasks[tasks["id"] == task_id]["setting"].iloc[0])
    game_plan = pd.read_csv(StringIO(task["fields"]),
                            names=range(0, len(task["fields"].split(";")[0].split("|"))),
                            sep="|",
                            lineterminator=";")  # rows and columns indexed from 1
    game_plan.index = range(0, len(task["fields"].split(";")))
    length = 10000
    energy = 10000
    if "length" in task:
        length = task["length"]
    if "energy" in task:
        energy = task["energy"]
    print(game_plan)
    return game_plan, length, energy


def is_crashed(game_plan, row_pos, col_pos):
    if row_pos < 0 or row_pos > len(game_plan.values) - 1:
        return True
    if col_pos < 0 or col_pos > len(game_plan.values[0]) - 1:
        return True
    if "A" in game_plan.values[col_pos][row_pos]:
        return True
    if "M" in game_plan.values[col_pos][row_pos]:
        return True
    return False


def is_completed(game_plan, row_pos, col_pos):
    if "b" in game_plan.values[row_pos][col_pos]:
        return True
    return False


def search_in_game_plan(content, game_plan, first=False):
    results = []
    for row in range(len(game_plan.values) - 1, -1, -1):
        for col in range(len(game_plan.values[0])):
            if content in game_plan.values[row][col]:
                results.append((row, col))
    print("Start at {}.".format(results[0]))
    if first:
        return results[0]
    else:
        return results


def shoot_meteoroid(row_pos, col_pos, game_plan):
    for results in search_in_game_plan("M", game_plan):
        if results[1] == row_pos:
            if results[0] > col_pos:
                game_plan[results[1]][results[0]] = 1
                return game_plan
    return game_plan


def load_operator_and_test_variable(pointer, program):
    operator = None
    test_position = None
    test_color = None

    if program[pointer + 1] == "x":
        mode = "position"
    else:
        mode = "color"
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
            if program[pointer + 3] == ">":
                operator = ">"
            elif program[pointer + 3] == "<":
                operator = "<"
            elif program[pointer + 3] == "=":
                operator = "=="
        # loading test_position
        if program[begin_pointer + 1] in ["0123456789"]:
            test_position = 10 * program[begin_pointer + 1] + program[
                begin_pointer + 2]  # operators, number+begin_pointer, number, {
            begin_pointer += 3  # operators, number, number, {, begin_pointer
        else:
            test_position = program[begin_pointer + 1]  # operators, number+begin_pointer, {
            begin_pointer += 2  # operators, number, {, begin_pointer
        # loading body of If statement
        end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer
        while program[end_pointer] != "}":
            end_pointer += 1
        end_pointer -= 1  # { begin_pointer, ..., end_pointer, }

    else:
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
        end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer
        while program[end_pointer] != "}":
            end_pointer += 1
        end_pointer -= 1  # { begin_pointer, ..., end_pointer, }

    return mode, operator, test_position, test_color, begin_pointer, end_pointer


def test(mode, operator, test_position, test_color, game_plan):
    if mode == "position":
        return eval("row_pos {} {}".format(operator, str(test_position)))
    else:
        return eval("game_plan.values[col_pos][row_pos][0] {} test_color".format(operator))


def forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer):
    row_pos -= 1
    square_sequence.append((row_pos, col_pos))
    if is_crashed(game_plan, row_pos, col_pos):
        correct = False
    pointer += 1
    return row_pos, col_pos, correct, square_sequence, pointer


def left(row_pos, col_pos, game_plan, correct, square_sequence, pointer):
    row_pos -= 1
    col_pos -= 1
    square_sequence.append((row_pos, col_pos))
    if is_crashed(game_plan, row_pos, col_pos):
        correct = False
    pointer += 1
    return row_pos, col_pos, correct, square_sequence, pointer


def right(row_pos, col_pos, game_plan, correct, square_sequence, pointer):
    row_pos -= 1
    col_pos += 1
    square_sequence.append((row_pos, col_pos))
    if is_crashed(game_plan, row_pos, col_pos):
        correct = False
    pointer += 1
    return row_pos, col_pos, correct, square_sequence, pointer


def shoot(row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy):
    game_plan = shoot_meteoroid(row_pos, col_pos, game_plan)
    row_pos, col_pos, correct, square_sequence, pointer = forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer)
    energy -= 1
    if energy < 0:
        correct = False
    return row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy


def repeat(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program):
    # loading number of repetitions
    if program[pointer + 2] in ["0123456789"]:
        repetitions = 10 * program[pointer + 1] + program[pointer + 2]
        begin_pointer = pointer + 4  # pointer, number, number, {, begin_pointer
    else:
        repetitions = program[pointer + 1]
        begin_pointer = pointer + 3  # pointer, number, {, begin_pointer
    # loading body of Repeat statement
    end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer
    while program[end_pointer] != "}":
        end_pointer += 1
    end_pointer -= 1  # { begin_pointer, ..., end_pointer, }

    for _ in range(repetitions):
        row_pos, col_pos, game_plan, correct, square_sequence = run_instructions(program[begin_pointer, end_pointer + 1],
                                                                             row_pos, col_pos, game_plan, correct,
                                                                             square_sequence)
    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2


def if_else(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program):
    mode, operator, test_position, test_color, begin_pointer, end_pointer = load_operator_and_test_variable(pointer, program)

    # loading Else part
    if program[end_pointer + 2] == "/":
        else_part = True
        else_begin_pointer = end_pointer + 4  # end_pointer, }, / {, else_begin_pointer
        else_end_pointer = else_begin_pointer + 1  # { else_begin_pointer, else_end_pointer
        while program[else_end_pointer] != "}":
            else_end_pointer += 1
        else_end_pointer -= 1  # { else_begin_pointer, ..., else_end_pointer, }
    else:
        else_part = False

    # execution
    if test(mode, operator, test_position, test_color, game_plan):
        row_pos, col_pos, game_plan, correct, square_sequence = run_instructions(program[begin_pointer, end_pointer + 1], row_pos, col_pos, game_plan, correct, square_sequence)
    else:
        if else_part:
            row_pos, col_pos, game_plan, correct, square_sequence = run_instructions(program[else_begin_pointer, else_end_pointer + 1], row_pos, col_pos, game_plan, correct, square_sequence)
            end_pointer = else_end_pointer

    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2


def while_robo(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program):
    mode, operator, test_position, test_color, begin_pointer, end_pointer = load_operator_and_test_variable(pointer, program)

    while test(mode, operator, test_position, test_color, game_plan):
        row_pos, col_pos, game_plan, correct, square_sequence = run_instructions(program[begin_pointer, end_pointer + 1], row_pos, col_pos, game_plan, correct, square_sequence)

    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2


def run_instructions(program, row_pos, col_pos, game_plan, correct, square_sequence, energy):
    pointer = 0

    while pointer < len(program):
        print(pointer, len(program)-1)
        if program[pointer] == "f":
            row_pos, col_pos, correct, square_sequence, pointer = forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer)
        elif program[pointer] == "l":
            row_pos, col_pos, correct, square_sequence, pointer = left(row_pos, col_pos, game_plan, correct, square_sequence, pointer)
        elif program[pointer] == "r":
            row_pos, col_pos, correct, square_sequence, pointer = right(row_pos, col_pos, game_plan, correct, square_sequence, pointer)
        elif program[pointer] == "s":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy = shoot(row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy)
        elif program[pointer] == "R":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer = repeat(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program)
        elif program[pointer] == "I":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer = if_else(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program)
        elif program[pointer] == "W":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer = while_robo(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program)
        else:
            raise(Exception, "Unknown symbol to process: {}".format(program[pointer]))
        if correct == False:
            return row_pos, col_pos, game_plan, correct, square_sequence
        if wormhole:
            row_pos, col_pos = wormhole_pos

    return row_pos, col_pos, game_plan, correct, square_sequence


#flrsRI/Wxyrbkg><=!0123456789{}
def run_task(tasks_path, task_id, program):
    program = re.sub("r{", "d{", program)
    game_plan, length, energy = load_game_plan(tasks_path=tasks_path, task_id=task_id)
    row_pos, col_pos = search_in_game_plan("S", game_plan, True)
    square_sequence = [(row_pos, col_pos)]
    correct = None

    if not program or len(program) > length:
        return False, square_sequence

    row_pos, col_pos, game_plan, correct, square_sequence = run_instructions(program, row_pos, col_pos, game_plan, correct, square_sequence, energy)

    if correct == False:
        print(False, square_sequence)
        return False, square_sequence
    if is_completed(game_plan, row_pos, col_pos):
        print(True, square_sequence)
        return True, square_sequence


# TODO: wormholes, diamonds
run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
         task_id=51,
         program="ffl")