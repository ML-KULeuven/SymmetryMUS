import math

import cpmpy as cp

def get_php_model(n_pigeons, n_holes):
    """
        Returns an instance of the pigon-hole problem
    """
    x = cp.boolvar(shape=(n_pigeons, n_holes), name="x")

    model = cp.Model()

    for pigeon in range(n_pigeons):
        model += cp.sum(x[pigeon]) >= 1

    for hole in range(n_holes):
        model += cp.sum(x[:, hole]) <= 1

    return model


def get_queens_model(dim, n_extra):
    """
        Returns an instance of the Boolean-encoded N+k-queens model.
    """
    board = cp.boolvar(shape=(dim, dim), name="board")

    model = cp.Model()
    model += cp.sum(board.flatten()) == (dim + n_extra)
    model += [cp.sum(row) <= 1 for row in board]
    model += [cp.sum(col) <= 1 for col in board.T]

    for offset in range(-dim + 1, dim):
        m += cp.sum(np.diagonal(board, offset=offset)) <= 1
        m += cp.sum(np.diagonal(np.fliplr(board), offset=offset)) <= 1

    return model

def get_binpacking_model(n_items, capacity, lb, ub, restrict_opt, seed=0):
    """
        Generate an unsatisfiable instance of the bin-packing problem.
        :param: n_items: the number of items to put in bins
        :param: capacity: the capacity of each bin
        :param: lb: the lower-bound for the each weight
        :param: ub: the upper-bound for each weight
        :param: restrict_opt: the fraction of the optimal number of bins
    """
    random.seed(seed)

    weights = [random.randint(lb,ub) for _ in range(n_items)]
    bv = cp.boolvar(shape=(n_items, n_items), name="bv") # row for each item, col for each bin
    is_used = cp.boolvar(shape=n_items)
    
    # objective
    model += [cp.any(col).implies(is_used[c]) for c,col in enumerate(bv.T)]
    # max capacity of each bin
    model += [cp.sum(col * weights) <= capacity for col in bv.T]
    # each item is in a bin
    model += [cp.sum(row) >= 1 for row in bv]
    model.minimize(cp.sum(is_used))

    if "gurobi" in cp.SolverLookup.solvernames():
        assert m.solve(solver="gurobi", Threads=1)
    else:
        assert m.solve(solver="ortools", num_workers=1)

    assert restrict_opt > 1 # ensure it is a percentage
    factor = restrict_opt / 100

    model += cp.sum(is_used) <  < math.floor(factor * s.objective_value())
    model.objective_ = None

    return model
