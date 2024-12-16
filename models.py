
import cpmpy as cp

def get_php_model(n_pigeons, n_holes):

    x = cp.boolvar(shape=(n_pigeons, n_holes), name="x")

    model = cp.Model()

    for pigeon in range(n_pigeons):
        model += cp.sum(x[pigeon]) >= 1

    for hole in range(n_holes):
        model += cp.sum(x[:, hole]) <= 1

    return model


def get_queens_model(dim, n_queens):

    x = cp.boolvar(shape=(dim, dim), name="x")

    model = cp.Model()

    for row in x:
        model += cp.sum(row) <= 1
    for col in x.T:
        model += cp.sum(col) <= 1

    # diagonals
    # TODO..