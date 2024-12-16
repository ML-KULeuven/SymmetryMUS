import itertools
import random
from time import time

import numpy as np

from cpmpy.expressions.core import Expression
from cpmpy.transformations.get_variables import get_variables
from natsort import natsorted


class Symmetry:

    def get_variables(self):
        raise NotImplementedError(f"Method `get_variables()` not implemented for type {type(self)}")

    def convert_to_assumptions(self, assump_map):
        raise NotImplementedError(f"Method `convert_to_assumptions()` not implemented for type {type(self)}")

class Permutation(Symmetry):
    """
        Permutation desribing a mapping of CPMpy variables to other CPMpy variables.
    """
    def __init__(self, permutation):
        """
            Initialized using disjoint cycle-notation.
            `permutation` is a list of tuples.
            E.g., (a,b,c)(d,e) describes the permutation mapping a to b, b to c and c to a; and d to e and e to d.
        """
        super(Permutation, self).__init__()

        assert isinstance(permutation, list), f"Expected list of tuples, got {type(permutation)}"
        for tup in permutation:
            assert isinstance(tup, tuple), f"Expected tuple, got {type(tup)}"

        self.perm = [tup for tup in permutation]
        self._fill_map()

    def _fill_map(self):
        self.map = dict()
        for tup in self.perm:
            for i, x in enumerate(tup):
                assert x not in self.map, "Duplicate "
                if i == len(tup) - 1:
                    self.map[str(x)] = tup[0]
                else:
                    self.map[str(x)] = tup[i + 1]

    def get_variables(self):
        return get_variables(self.perm)

    def __str__(self):
        return ",".join([str(tup) for tup in self.perm])

    def __getitem__(self, item):
        return self.map.get(item, item)

    def get_symmetric_images_in_subset(self, subset, var, n_tries=-1):
        lst_subset = natsorted(subset, key=str)
        try:
            idx = next(i for i, x in enumerate(lst_subset) if hash(x) == hash(var))
        except StopIteration: # var not described in this permutation
            return frozenset({var})

        new_lst = lst_subset
        full_images = set()
        valid_images = []
        n = 0
        while n != n_tries:
            new_lst = tuple([self[x] for x in new_lst])
            if new_lst in full_images: # wrapped around
                break
            if set(new_lst) == subset:
                valid_images.append(new_lst[idx])
            full_images.add(new_lst)
            n += 1

        return frozenset(valid_images)


    def apply_symmetry(self, subset, n=-1, exclude=set()):
        """
           Find symmetric counterparts of `subset`, including `subset` itself
           Works by repeatedly applying the mapping until fixpoint
           Do not count symmetric images already found
       """
        init = frozenset(subset)
        subset = init
        images = {init}

        if subset not in exclude:
            n_generated = 1
            yield subset
        else:
            n_generated = 0

        while n_generated != n:
            subset = frozenset(self[x] for x in subset)
            if subset in images:
                break
            images.add(subset)
            if subset not in exclude:
                n_generated += 1
                yield subset

    def convert_to_assumptions(self, assump_map):
        for i, tup in enumerate(self.perm):
            self.perm[i] = tuple([assump_map[cons] for cons in tup])
        self._fill_map()

class RowSymmetry(Symmetry):

    def __init__(self, matrix):
        super(RowSymmetry, self).__init__()
        for row in matrix:
            for i, var in enumerate(row):
                assert isinstance(var, Expression), f"Expected matrix of all variables, got {type(var)} in row {i}"
        self.matrix = np.array(matrix)
        assert len(self.matrix.shape) == 2

    def get_variables(self):
        return get_variables(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def apply_symmetry(self, subset, n=-1, exclude=set()):
        """
            Generate the symmetric images of the given subset `subset`
        """
        subset = frozenset(subset)
        add_always = frozenset([v for v in subset if v not in set(self.matrix.flat)]) # vars not defined in the symmetry

        xs, ys = np.where(np.vectorize(lambda x: x in subset)(self.matrix))
        hit_rows = frozenset(xs)

        if subset not in exclude:
            found_images = {subset}
            yield subset
        else:
            found_images = set()

        for i,j in itertools.combinations(list(range(len(self.matrix))), 2):
            if len(found_images) == n:
                break

            if i in hit_rows or j in hit_rows: # swapping i and j
                new_subset = set()
                for x,y in zip(xs, ys):
                    if x == i:
                        new_subset.add(self.matrix[j,y])
                    elif x == j:
                        new_subset.add(self.matrix[i,y])
                    else:
                        new_subset.add(self.matrix[x,y])

                new_subset |= add_always
                new_subset = frozenset(new_subset)
                assert len(new_subset) == len(subset)

                if new_subset not in exclude and new_subset not in found_images:
                    found_images.add(new_subset)
                    yield new_subset

    def convert_to_assumptions(self, assump_map):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                self.matrix[i,j] = assump_map[self.matrix[i,j]]

    def get_symmetric_images_in_subset(self, subset, var):
        """
            Find the symmetric images of a variable wrt to a given subset
            I.e., find a symmetry described by the matrix whose image is again 'subset' and
                return the element 'var' maps to.
        """
        # find row of var
        for i,row in enumerate(self.matrix):
            if var in set(row):
                break
        else: # var is not in matrix
            return {var}

        var_i = i
        var_j = next(j for j,x in enumerate(self.matrix[var_i]) if str(x) == str(var))

        images = set()
        # columns to consider
        js = [j for j,x in enumerate(row) if x in subset]
        for i2, row2 in enumerate(self.matrix):
            # get indices of vars in row2
            js2 = [j for j,x in enumerate(row2) if x in subset]
            if js == js2: # swapping `row` with `row2` yields `subset` again!
                images.add(row2[var_j])

        return images
