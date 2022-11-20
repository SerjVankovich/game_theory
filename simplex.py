import math

import numpy as np

MIN_VALUE: list = [0, 0]


def create_simplex_table(c: np.ndarray, b: np.ndarray, A: np.ndarray) -> np.ndarray:
    table: np.ndarray = np.copy(A)
    try:
        table = np.hstack((b.reshape((len(b), 1)), table))

        last_stroke = np.hstack((np.array([0]), c, np.zeros(len(A[0]) - len(c))))
        table = np.vstack((table, last_stroke))
    except Exception:
        print('Error create table')
    return table


def print_table(table: np.ndarray) -> None:
    for row in table:
        for col in row:
            print(f"%.2f" % col, end="    ")
        print()
    print()


def create_variables(basis: list, table: np.ndarray) -> list:
    answer: list = [0] * len(basis)
    for i, x in enumerate(basis):
        if x != -1 and x < len(basis):
            answer[x] = table[i, 0]
    return answer


def simplex_method(table: np.ndarray) -> (np.ndarray, list, float):
    table_to_work = np.copy(table)
    basis = [-1] * (len(table_to_work) - 1)
    while True:
        min_col: int = np.argmin(table_to_work[-1]).min()
        if table_to_work[-1, min_col] >= 0:
            return table_to_work, create_variables(basis, table_to_work), table_to_work[-1, 0]
        min_row = find_min_row(min_col, table_to_work)
        basis[min_row] = min_col - 1
        table_to_work[min_row] /= table_to_work[min_row, min_col]
        for i in range(table_to_work.shape[0]):
            if i != min_row:
                table_to_work[i] -= table_to_work[min_row] * table_to_work[i, min_col]


def find_min_row(min_col, table_to_work):
    min_elem = float('inf')
    min_stroke = 0
    for i in range(len(table_to_work) - 1):
        if table_to_work[i, min_col] > 0:
            div = table_to_work[i, 0] / table_to_work[i, min_col]
            if div < min_elem:
                min_elem = div
                min_stroke = i
    return min_stroke


def reformat_table(table: np.ndarray, b, x_index, fic_x) -> np.ndarray:
    result = np.copy(table)
    result = np.hstack((result, np.zeros((len(table), 1))))
    row_to_add = np.zeros(len(table[0]) + 1)
    row_to_add[0] = b
    row_to_add[x_index + 1] = 1
    row_to_add[-1] = fic_x
    result = np.vstack((result[:-1], row_to_add, result[-1]))
    return result


def branches_and_bounds(table: np.ndarray):
    global MIN_VALUE
    table1, answer, estimation = simplex_method(table)
    print(answer)
    print(MIN_VALUE)
    print(estimation)
    print()
    if int(estimation) <= MIN_VALUE[1]:
        return
    isInteger: bool = True
    for i, x in enumerate(answer):
        if x - int(x) > 10e-4:
            first_table = reformat_table(table, int(x), i, 1)
            second_table = reformat_table(table, int(x) + 1, i, -1)
            branches_and_bounds(first_table)
            branches_and_bounds(second_table)
            isInteger = False
    if isInteger and estimation > MIN_VALUE[1]:
        MIN_VALUE = [answer, estimation]


c = np.array([-2, -3])
b = np.array([24, 22])
A = np.array([
    [3, 4, 1, 0],
    [2, 5, 0, 1]
])

branches_and_bounds(create_simplex_table(c, b, A))
