import numpy as np


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


def find_min_col(min_row, table_to_work):
    min_elem = float('inf')
    min_index = 1
    for i in range(1, len(table_to_work[0])):
        if table_to_work[min_row, i] < 0:
            div = abs(table_to_work[-1, i] / table_to_work[min_row, i])
            if div < min_elem:
                min_elem = div
                min_index = i
    if min_elem == float('inf'):
        return -1
    return min_index


def simplex_method(table: np.ndarray) -> (np.ndarray, list, float):
    table_to_work = np.copy(table)
    basis = [-1] * (len(table_to_work) - 1)

    # first step
    while True:
        min_row: int = np.argmin(table_to_work[:, 0]).min()
        if table_to_work[min_row, 0] >= 0:
            break
        min_col: int = find_min_col(min_row, table_to_work)

        # have not valid solutions with restrictions
        if min_col == -1:
            return table_to_work, basis, float('-inf')
        simplex_step(basis, min_col, min_row, table_to_work)

    # second step
    while True:
        min_col: int = np.argmin(table_to_work[-1]).min()
        if table_to_work[-1, min_col] >= 0:
            return table_to_work, create_variables(basis, table_to_work), table_to_work[-1, 0]
        min_row = find_min_row(min_col, table_to_work)
        simplex_step(basis, min_col, min_row, table_to_work)


def simplex_step(basis, min_col, min_row, table_to_work):
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
    if fic_x == -1:
        row_to_add *= -1
    result = np.vstack((result[:-1], row_to_add, result[-1]))
    return result


def choose_table(all_branches: list) -> (np.ndarray, list):
    arg = np.argmax(np.array([x[2] for x in all_branches]))
    return all_branches.pop(arg)


def all_int(answer: list):
    for x in answer:
        if abs(x - int(x)) > 10e-6:
            return False
    return True


def branch_and_bounds(table: np.ndarray) -> (list, float):
    all_branches = []
    max_optimum = 0
    table1, answer, estimation = simplex_method(table)
    while True:
        for i, x in enumerate(answer):
            if abs(x - int(x)) > 10e-6:
                first_table = reformat_table(table, int(x), i, 1)
                second_table = reformat_table(table, int(x) + 1, i, -1)
                print("first table: \n", first_table)
                print("first x:", int(x))
                print()
                print("second table: \n", second_table)
                print("second x:", int(x) + 1)
                table1, answer1, estimation1 = simplex_method(first_table)
                table2, answer2, estimation2 = simplex_method(second_table)

                if estimation1 != float('inf') and estimation1 != float('-inf') and int(estimation1) > max_optimum:
                    all_branches.append((first_table, answer1, estimation1))
                if estimation2 != float('inf') and estimation2 != float('-inf') and int(estimation2) > max_optimum:
                    all_branches.append((second_table, answer2, estimation2))

                if all_int(answer1):
                    if estimation1 > max_optimum:
                        max_optimum = estimation1

                if all_int(answer2):
                    if estimation2 > max_optimum:
                        max_optimum = estimation2

                print("first branch:", answer1, estimation1)
                print()
                print("second branch:", answer2, estimation2)
                print()
                table, answer, estimation = choose_table(all_branches)
                print("chosen branch:", answer, estimation)
                if all_int(answer):
                    return table, answer, estimation
                print('\n', '\n')
                break


c_ = np.array([-2, -3])
b_ = np.array([24, 22])
A_ = np.array([[3, 4, 1, 0],
              [2, 5, 0, 1]])

table = create_simplex_table(c_, b_, A_)

_, ans, est = branch_and_bounds(table)
print(ans, est)
