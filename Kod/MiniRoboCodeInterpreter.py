import re
import InterpreterTools as it


def forward(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes, diamonds, steps, verbose):
    """
    Processes forward statement.
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
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds; updated input variables
    """
    row_pos -= 1
    row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
        it.check_position(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                          diamonds, steps, verbose)
    return row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps


def left(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes, diamonds, steps, verbose):
    """
    Processes left statement.
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
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds; updated input variables
    """
    row_pos -= 1
    col_pos -= 1
    row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
        it.check_position(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                          diamonds, steps, verbose)
    return row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps


def right(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes, diamonds, steps, verbose):
    """
    Processes right statement.
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
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds; updated input variables
    """
    row_pos -= 1
    col_pos += 1
    row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
        it.check_position(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                          diamonds, steps, verbose)
    return row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps


def shoot(row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, wormholes, diamonds, steps, verbose):
    """
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param pointer: int; position of processed character in program
    :param energy: int; shoots remaining
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds; updated input variables
    """
    steps -= 1
    game_board = it.shoot_meteoroid(row_pos, col_pos, game_board)
    row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
        forward(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes, diamonds, steps, verbose)
    energy -= 1
    return row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds, steps


def repeat(row_pos, col_pos, game_board, correct, square_sequence, pointer,
           program, energy, wormholes, diamonds, steps, verbose):
    """
    Processes repeat statement.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param pointer: int; position of processed character in program
    :param program: string; current program
    :param energy: int; shoots remaining
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds; updated input variables
    """
    steps -= 1

    # loading number of repetitions
    if program[pointer + 2] in "0123456789":
        repetitions = 10 * int(program[pointer + 1]) + int(program[pointer + 2])
        begin_pointer = pointer + 4  # pointer, number, number, {, begin_pointer
    else:
        repetitions = int(program[pointer + 1])
        begin_pointer = pointer + 3  # pointer, number, {, begin_pointer
    # loading body of Repeat statement
    if program[begin_pointer] == "}":  # operator(s), {, }+begin_pointer
        end_pointer = begin_pointer - 1
        return row_pos, col_pos, game_board, correct, square_sequence, end_pointer + 2, energy, diamonds, steps

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
        if steps > 0 and correct != False:
            row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps = \
                run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                                 program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds, steps, verbose)

    return row_pos, col_pos, game_board, correct, square_sequence, end_pointer + 2, energy, diamonds, steps


def if_else(row_pos, col_pos, game_board, correct, square_sequence, pointer,
            program, energy, wormholes, diamonds, steps, verbose):
    """
    Processes if(-else) statement.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param pointer: int; position of processed character in program
    :param program: string; current program
    :param energy: int; shoots remaining
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds; updated input variables
    """
    mode, operator, test_position, test_color, begin_pointer, end_pointer = \
        it.load_operator_and_test_variable(pointer, program)
    steps -= 1

    # loading Else part
    if end_pointer + 2 < len(program):
        if program[end_pointer + 2] == "/":
            else_part = True
            else_begin_pointer = end_pointer + 4  # end_pointer, }, / {, else_begin_pointer
            if program[else_begin_pointer] == "}":  # /, {, }+else_begin_pointer
                else_end_pointer = else_begin_pointer - 1
            else:
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
    if mode is None:
        if else_end_pointer:
            end_pointer = else_end_pointer
    else:
        if it.test(mode, operator, test_position, test_color, row_pos, col_pos, game_board):
            row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps = \
                run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                                 program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds, steps, verbose)
            if else_end_pointer:
                end_pointer = else_end_pointer
        else:
            if else_part:
                row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps = \
                    run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                                     program[else_begin_pointer: else_end_pointer + 1], energy, wormholes,
                                     diamonds, steps, verbose)
                end_pointer = else_end_pointer

    return row_pos, col_pos, game_board, correct, square_sequence, end_pointer + 2, energy, diamonds, steps


def while_robo(row_pos, col_pos, game_board, correct, square_sequence, pointer,
               program, energy, wormholes, diamonds, steps, verbose):
    """
    Processes while statement.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param pointer: int; position of processed character in program
    :param program: string; current program
    :param energy: int; shoots remaining
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds; updated input variables
    """
    mode, operator, test_position, test_color, begin_pointer, end_pointer = \
        it.load_operator_and_test_variable(pointer, program)

    if mode is not None:
        while steps > 0 and correct != False and it.test(mode, operator, test_position, test_color, row_pos, col_pos, game_board):
            steps -= 1
            row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps = \
                run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                                 program[begin_pointer: end_pointer + 1], energy, wormholes, diamonds, steps, verbose)

    return row_pos, col_pos, game_board, correct, square_sequence, end_pointer + 2, energy, diamonds, steps


def run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                     program, energy, wormholes, diamonds, steps, verbose):
    """
    Interprets (sub)program.
    :param row_pos: int; row position
    :param col_pos: int; column position
    :param game_board: pd.DataFrame; game_board state
    :param correct: bool; correctness of the task session
    :param square_sequence: list of 2-int-tuples; sequence of visited squares
    :param program: string; current program
    :param energy: int; shoots remaining
    :param wormholes: dict of string: list of 2-int-tuples; wormholes positions
    :param diamonds: list of 2-int-tuples; positions of non-collected diamonds
    :param steps: int; number of remaining steps
    :param verbose: bool; verbosity
    :return: row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps; updated input variables
    """
    pointer = 0
    while pointer < len(program) and steps > 0 and correct != False:
        # print("""Pointer: {}\n Program: {}\n Position: ({}, {})\n Square sequence: {}\n Diamonds: {}\n{}
        #       """.format(pointer, program, row_pos, col_pos, square_sequence, diamonds, game_board))
        if program[pointer] == "f":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
                forward(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                        diamonds, steps, verbose)
        elif program[pointer] == "l":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
                left(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                     diamonds, steps, verbose)
        elif program[pointer] == "r":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
                right(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                      diamonds, steps, verbose)
        elif program[pointer] == "s":
            if energy > 0:
                row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds, steps = \
                    shoot(row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, wormholes,
                          diamonds, steps, verbose)
            else:
                row_pos, col_pos, game_board, correct, square_sequence, pointer, diamonds, steps = \
                    forward(row_pos, col_pos, game_board, correct, square_sequence, pointer, wormholes,
                            diamonds, steps, verbose)
        elif program[pointer] == "R":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds, steps = \
                repeat(row_pos, col_pos, game_board, correct, square_sequence, pointer,
                       program, energy, wormholes, diamonds, steps, verbose)
        elif program[pointer] == "I":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds, steps = \
                if_else(row_pos, col_pos, game_board, correct, square_sequence, pointer,
                        program, energy, wormholes, diamonds, steps, verbose)
        elif program[pointer] == "W":
            row_pos, col_pos, game_board, correct, square_sequence, pointer, energy, diamonds, steps = \
                while_robo(row_pos, col_pos, game_board, correct, square_sequence, pointer,
                           program, energy, wormholes, diamonds, steps, verbose)
        else:
            raise Exception("Unknown symbol to process: {}".format(program[pointer]))

    return row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps


# flrsRI/Wxyrbkg><=!0123456789{}
def run_task(tasks_path, task_id, program, verbose=False):
    """
    Runs selected task with given program.
    :param tasks_path: path to tasks.csv
    :param task_id: id of task played
    :param program: string; program to run
    :param verbose: bool; verbosity
    :return correct: bool; correctness of task session
    :return square_sequence: list of 2-int-tuples; sequence of visited squares
    """
    program = re.sub("r{", "d{", program)
    game_board, length, energy = it.load_game_board(tasks_path=tasks_path, task_id=task_id)
    row_pos, col_pos = it.search_in_game_board("S", game_board, True)
    square_sequence = [(row_pos, col_pos)]
    correct = None
    steps = 1000

    wormholes = {"wormhole_w": it.search_in_game_board("W", game_board),
                 "wormhole_x": it.search_in_game_board("X", game_board),
                 "wormhole_y": it.search_in_game_board("Y", game_board),
                 "wormhole_z": it.search_in_game_board("Z", game_board)}
    diamonds = it.search_in_game_board("D", game_board)

    if program == "nan":
        program = ""

    if verbose:
        print("\nTask_id: {}\nProgram: {}".format(task_id, program))
        print("Energy: {}\nLength: {}\n{}".format(energy, length, game_board))

    if not program or len(re.sub("[{}0123456789<>=!xbkygd/]", "", program)) > length:
        if verbose:
            print("Correct: {}".format(False))
            # print("Square sequence: {}\n{}\n".format(square_sequence, game_board))
        return False, square_sequence

    row_pos, col_pos, game_board, correct, square_sequence, energy, diamonds, steps = \
        run_instructions(row_pos, col_pos, game_board, correct, square_sequence,
                         program, energy, wormholes, diamonds, steps, verbose)
    if verbose:
        print("Steps remaining:", steps)
    if steps <= 0:
        if verbose:
            print("INCORRECT – steps limit reached")
        return False, square_sequence
    if correct == False:  # "if not correct" would succeed if correct was None
        if verbose:
            print("Correct: {}".format(False))
            # print("Square sequence: {}\n{}\n".format(square_sequence, game_board))
        return False, square_sequence
    if it.is_completed(row_pos, col_pos, game_board, diamonds, verbose):
        if verbose:
            print("Correct: {}".format(True))
            # print("Square sequence: {}\n{}\n".format(square_sequence, game_board))
        return True, square_sequence
    if verbose:
        print("Correct: {}".format(False))
        # print("Square sequence: {}\n{}\n".format(square_sequence, game_board))
    return False, square_sequence





import random
from Tools import load_extended_snapshots

submits = load_extended_snapshots(snapshots_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/program_snapshots.csv",
                                  task_sessions_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/task_sessions.csv",
                                  tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                                  task_sessions_cols=["id", "task"],
                                  tasks_cols=["id", "setting", "solution"])
submits = submits[submits["granularity"] == "execution"]
submits = submits[["id", "program", "correct", "task"]]
#print(len(submits["id"].values))
test_ids = submits["id"].values[10000:20000]
#test_ids = [10, 181, 298, 381, 491, 691, 694, 1105, 1276, 1355, 1761, 2093, 2177, 2179, 2389, 2483, 2721, 2845, 3132, 3317, 4714, 5021, 5240, 5296, 5812, 6867, 7156, 7357, 7760, 7883, 7962, 8013, 8198, 8232, 8250, 9143, 9293, 9298, 9393, 9635, 9714, 9901, 9920, 9975, 9977, 9993, 10139, 10683, 10796, 10813, 11248, 11645, 11705, 11865, 12528, 12976, 13504, 13538, 14591, 15132, 15152, 15313, 15521, 16166, 16517, 16518, 16519, 16542, 16543, 16735, 17005, 17094, 17472, 17533, 18620, 18805, 19050, 19147, 20385, 21884, 21951, 22784, 23244, 23621, 23836, 24327, 24353, 24578, 24754, 24891, 25099, 25202, 25358, 25518, 25750, 25848, 26239, 26431, 26527, 26547, 26748, 26762, 26828, 27875, 27893, 28024, 28353, 28889, 29100, 29390, 29941, 30468, 30630, 31521, 31794, 32605, 32606, 32748, 32913, 33246, 33344, 33485, 33768, 34032, 34082, 35183, 35317, 35599, 35913, 36038, 36057, 36090, 36123, 36214, 36364, 36480, 36570, 36648, 36766, 36948, 37684, 38229, 38352, 39009, 39202, 39248, 39495, 40624, 40960, 41406, 42281, 42286, 42353, 42740, 42790, 42847, 42896, 43083, 43941, 43994, 44443, 44586, 44837, 44856, 45259, 45724, 45914, 46144, 46292]
#print(len(test_ids))
wrong = []
#for i in range(0):
#    test_ids.append(random.choice(submits["id"].values))
for test_id in test_ids:
    print(test_id)
    task_id = int(submits[submits["id"] == test_id]["task"].values[0])
    program = str(submits[submits["id"] == test_id]["program"].values[0])
    correct = bool(submits[submits["id"] == test_id]["correct"].values[0])
    #print(test_id, task_id, program, correct)
    run_correct, _ = run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
                              task_id=task_id,
                              program=program,
                              verbose=False)
    agreement = correct == run_correct
    #print("AGREEMENT: {}\n\n".format(agreement))
    if not agreement:
        wrong.append(test_id)
print("Non-agree test_ids:", wrong)

"""
import pandas as pd
task_data = pd.read_csv("C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv")
for i in range(0, len(task_data.index)):
    run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
             task_id=int(task_data.iloc[i]["id"]),
             program=task_data.iloc[i]["solution"],
             verbose=True)
"""
"""
run_task(tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-09-08/tasks.csv",
         task_id=45,
         program="R2{W!d{r}W!d{l}}W!b{r}",
         verbose=True)
"""
