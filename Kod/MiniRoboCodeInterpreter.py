import ast
from io import StringIO
import pandas as pd
import random
import re


def load_game_plan(tasks_path, task_id):
    tasks = pd.read_csv(tasks_path)
    task = ast.literal_eval(tasks[tasks["id"] == task_id]["setting"].iloc[0])
    game_plan = pd.read_csv(StringIO(re.sub("r", "d", task["fields"])),
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
    return game_plan, length, energy


def is_crashed(row_pos, col_pos, game_plan):
    if row_pos < 0 or row_pos > len(game_plan.values) - 1:
        print("CRASHED - Out of game plan rows, ({}, {})".format(row_pos, col_pos))
        return True
    if col_pos < 0 or col_pos > len(game_plan.values[0]) - 1:
        print("CRASHED - Out of game plan cols, ({}, {})".format(row_pos, col_pos))
        return True
    if "A" in game_plan.values[row_pos][col_pos]:
        print("CRASHED - Crashed to asteroid, ({}, {})".format(row_pos, col_pos))
        return True
    if "M" in game_plan.values[row_pos][col_pos]:
        print("CRASHED - Crashed to meteoroid, ({}, {})".format(row_pos, col_pos))
        return True
    return False


def is_completed(game_plan, row_pos, col_pos, diamonds):
    if "b" in game_plan.values[row_pos][col_pos]:
        if len(diamonds) == 0:
            return True
        else:
            print("INCORRECT - Did not collect all diamonds. Diamonds: {}".format(diamonds))
    else:
        print("INCORRECT - not on the finish line")
    return False


def check_position(row_pos, col_pos, game_plan, correct, square_sequence, wormholes, diamonds):
    if (row_pos, col_pos) in diamonds:
        game_plan[col_pos][row_pos] = re.sub("D", "", game_plan[col_pos][row_pos])
        diamonds = list(set(diamonds) - {(row_pos, col_pos)})
    for wormhole_type in wormholes:
        if (row_pos, col_pos) in wormholes[wormhole_type]:
            row_pos, col_pos = random.choice(list(set(wormholes[wormhole_type]) - {(row_pos, col_pos)}))
            square_sequence.append((row_pos, col_pos))
    if (row_pos, col_pos) in diamonds:
        game_plan[col_pos][row_pos] = re.sub("D", "", game_plan[col_pos][row_pos])
        diamonds = list(set(diamonds) - {(row_pos, col_pos)})
    if is_crashed(row_pos, col_pos, game_plan):
        correct = False
    return row_pos, col_pos, game_plan, correct, square_sequence, diamonds


def search_in_game_plan(content, game_plan, first=False):
    results = []
    for row in range(len(game_plan.values) - 1, -1, -1):
        for col in range(len(game_plan.values[0])):
            if content in game_plan.values[row][col]:
                results.append((row, col))
    if first:
        return results[0]
    else:
        return results


def shoot_meteoroid(row_pos, col_pos, game_plan):
    for result in search_in_game_plan("M", game_plan):
        if result[1] == col_pos:
            if result[0] < row_pos:
                game_plan[result[1]][result[0]] = re.sub("M", "", game_plan[result[1]][result[0]])
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
            if program[pointer + 2] == ">":
                operator = ">"
            elif program[pointer + 2] == "<":
                operator = "<"
            elif program[pointer + 2] == "=":
                operator = "=="
        # loading test_position
        if program[begin_pointer + 1] in ["0123456789"]:
            test_position = 10 * int(program[begin_pointer]) + int(program[begin_pointer + 1])  # operators, number+begin_pointer, number, {
            test_position -= 1  # RoboMission counts from 1
            begin_pointer += 3  # operators, number, number, {, begin_pointer
        else:
            test_position = int(program[begin_pointer])  # operators, number+begin_pointer, {
            test_position -= 1  # RoboMission counts from 1
            begin_pointer += 2  # operators, number, {, begin_pointer
        # loading body of If statement
        end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer
        foreign_parentheses = 0
        while program[end_pointer] != "}" or foreign_parentheses != 0:
            if program[end_pointer] == "{":
                foreign_parentheses += 1
            elif program[end_pointer] == "}":
                foreign_parentheses -= 1
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
        foreign_parentheses = 0
        while program[end_pointer] != "}" or foreign_parentheses != 0:
            if program[end_pointer] == "{":
                foreign_parentheses += 1
            elif program[end_pointer] == "}":
                foreign_parentheses -= 1
            end_pointer += 1
        end_pointer -= 1  # { begin_pointer, ..., end_pointer, }
    return mode, operator, test_position, test_color, begin_pointer, end_pointer


def test(mode, operator, test_position, test_color, row_pos, col_pos, game_plan):
    if mode == "position":
        return eval("col_pos {} {}".format(operator, str(test_position)))
    else:
        return eval("game_plan.values[row_pos][col_pos][0] {} test_color".format(operator))


def forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds):
    row_pos -= 1
    square_sequence.append((row_pos, col_pos))
    pointer += 1
    row_pos, col_pos, game_plan, correct, square_sequence, diamonds = check_position(row_pos, col_pos, game_plan, correct, square_sequence, wormholes, diamonds)
    return row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds


def left(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds):
    row_pos -= 1
    col_pos -= 1
    square_sequence.append((row_pos, col_pos))
    pointer += 1
    row_pos, col_pos, game_plan, correct, square_sequence, diamonds = check_position(row_pos, col_pos, game_plan, correct, square_sequence, wormholes, diamonds)
    return row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds


def right(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds):
    row_pos -= 1
    col_pos += 1
    square_sequence.append((row_pos, col_pos))
    pointer += 1
    row_pos, col_pos, game_plan, correct, square_sequence, diamonds = check_position(row_pos, col_pos, game_plan, correct, square_sequence, wormholes, diamonds)
    return row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds


def shoot(row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, wormholes, diamonds):
    game_plan = shoot_meteoroid(row_pos, col_pos, game_plan)
    row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds = forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds)
    energy -= 1
    if energy < 0:
        correct = False
    return row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, diamonds


def repeat(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds):
    # loading number of repetitions
    if program[pointer + 2] in ["0123456789"]:
        repetitions = 10 * int(program[pointer + 1]) + int(program[pointer + 2])
        begin_pointer = pointer + 4  # pointer, number, number, {, begin_pointer
    else:
        repetitions = int(program[pointer + 1])
        begin_pointer = pointer + 3  # pointer, number, {, begin_pointer
    # loading body of Repeat statement
    end_pointer = begin_pointer + 1  # { begin_pointer, end_pointer

    foreign_parentheses = 0
    while program[end_pointer] != "}" or foreign_parentheses != 0:
        if program[end_pointer] == "{":
            foreign_parentheses += 1
        elif program[end_pointer] == "}":
            foreign_parentheses -= 1
        end_pointer += 1
    end_pointer -= 1  # { begin_pointer, ..., end_pointer, }

    for _ in range(repetitions):
        row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds = run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds)

    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2, energy, diamonds


def if_else(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds):
    mode, operator, test_position, test_color, begin_pointer, end_pointer = load_operator_and_test_variable(pointer, program)

    # loading Else part
    if end_pointer + 2 < len(program):
        if program[end_pointer + 2] == "/":
            else_part = True
            else_begin_pointer = end_pointer + 4  # end_pointer, }, / {, else_begin_pointer
            else_end_pointer = else_begin_pointer + 1  # { else_begin_pointer, else_end_pointer
            foreign_parentheses = 0
            while program[else_end_pointer] != "}" or foreign_parentheses != 0:
                if program[else_end_pointer] == "{":
                    foreign_parentheses += 1
                elif program[else_end_pointer] == "}":
                    foreign_parentheses -= 1
                else_end_pointer += 1
            else_end_pointer -= 1  # { else_begin_pointer, ..., else_end_pointer, }
        else:
            else_part = False
            else_end_pointer = None
    else:
        else_part = False
        else_end_pointer = None
    # execution
    if test(mode, operator, test_position, test_color, row_pos, col_pos, game_plan):
        row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds = run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds)
        if else_end_pointer:
            end_pointer = else_end_pointer
    else:
        if else_part:
            row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds = run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program[else_begin_pointer: else_end_pointer + 1], energy, wormholes, diamonds)
            end_pointer = else_end_pointer

    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2, energy, diamonds


def while_robo(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds):
    mode, operator, test_position, test_color, begin_pointer, end_pointer = load_operator_and_test_variable(pointer, program)

    while test(mode, operator, test_position, test_color, row_pos, col_pos, game_plan):
        row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds = run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds)

    return row_pos, col_pos, game_plan, correct, square_sequence, end_pointer + 2, energy, diamonds


def run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program, energy, wormholes, diamonds):
    pointer = 0

    while pointer < len(program):
        #print("Pointer: {}\n Program: {}\n Position: ({}, {})\n Square sequence: {}\n Diamonds: {}\n{}".format(pointer, program, row_pos, col_pos, square_sequence, diamonds, game_plan))
        if program[pointer] == "f":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds = forward(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds)
        elif program[pointer] == "l":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds = left(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds)
        elif program[pointer] == "r":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, diamonds = right(row_pos, col_pos, game_plan, correct, square_sequence, pointer, wormholes, diamonds)
        elif program[pointer] == "s":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, diamonds = shoot(row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, wormholes, diamonds)
        elif program[pointer] == "R":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, diamonds = repeat(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds)
        elif program[pointer] == "I":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, diamonds = if_else(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds)
        elif program[pointer] == "W":
            row_pos, col_pos, game_plan, correct, square_sequence, pointer, energy, diamonds = while_robo(row_pos, col_pos, game_plan, correct, square_sequence, pointer, program, energy, wormholes, diamonds)
        else:
            raise Exception("Unknown symbol to process: {}".format(program[pointer]))

    return row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds


#flrsRI/Wxyrbkg><=!0123456789{}
def run_task(tasks_path, task_id, program):
    program = re.sub("r{", "d{", program)
    game_plan, length, energy = load_game_plan(tasks_path=tasks_path, task_id=task_id)
    row_pos, col_pos = search_in_game_plan("S", game_plan, True)
    square_sequence = [(row_pos, col_pos)]
    correct = None

    wormholes = {"wormhole_w": search_in_game_plan("W", game_plan),
                 "wormhole_x": search_in_game_plan("X", game_plan),
                 "wormhole_y": search_in_game_plan("Y", game_plan),
                 "wormhole_z": search_in_game_plan("Z", game_plan)}
    diamonds = search_in_game_plan("D", game_plan)

    print("\nTask_id: {}\nProgram: {})".format(task_id, program))
    #print("Energy: {}\nLength: {}\n{}".format(energy, length, game_plan))

    if not program or len(re.sub("[{}0123456789<>=!xbkygd/]", "", program)) > length:
        print("Correct: {}".format(False))
        #print("Square sequence: {}\n{}\n".format(square_sequence, game_plan))
        return False, square_sequence
    row_pos, col_pos, game_plan, correct, square_sequence, energy, diamonds = run_instructions(row_pos, col_pos, game_plan, correct, square_sequence, program, energy, wormholes, diamonds)

    if correct == False:
        print("Correct: {}".format(False))
        #print("Square sequence: {}\n{}\n".format(square_sequence, game_plan))
        return False, square_sequence
    if is_completed(game_plan, row_pos, col_pos, diamonds):
        print("Correct: {}".format(True))
        #print("Square sequence: {}\n{}\n".format(square_sequence, game_plan))
        return True, square_sequence
    print("Correct: {}".format(False))
    #print("Square sequence: {}\n{}\n".format(square_sequence, game_plan))
    return False, square_sequence


task_data = pd.read_csv("C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")
for i in range(0, len(task_data.index)):
    run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
             task_id=int(task_data.iloc[i]["id"]),
             program=task_data.iloc[i]["solution"])

"""
run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
         task_id=80,
         program="W!b{Ix=1{f}/{l}}")
"""
