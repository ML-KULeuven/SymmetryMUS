
"""
    This file implements a Python wrapper for the BreakID CNF symmetry breaking tool
"""
import os

import numpy as np
import cpmpy as cp

from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.to_cnf import to_cnf
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.reification import reify_rewrite, only_implies, only_bv_reifies
from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.linearize import linearize_constraint, only_positive_bv

from cpmpy.expressions.variables import _NumVarImpl, _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.utils import is_num, get_bounds

import subprocess
from subprocess import Popen, PIPE, run, STDOUT
import tempfile

from natsort import natsorted

from symmetries import RowSymmetry, Permutation

BREAKID_PATH = "/usr/local/breakid/src/BreakID"

class BreakID:
    """
        Python/CPMpy wrapper for the BreakID tool.
        BreakID can be downloaded from bitbucket at the following url:
            https://bitbucket.org/krr/breakid/src/master/
    """
    def __init__(self, path_to_binary=BREAKID_PATH):
        """
            Initialize the breakid wrapper
            :param path_to_binary: path to the BreakID executable
        """
        self.path_to_binary = path_to_binary
        self.verbose = True

    supported_kwargs = { # map of BreakID parameters and their type as used in CLI
        "f":str, "no-row":bool, "no-bin":bool, "no-small":bool, "no-relaxed":bool,
        "s":int, "t":int, "v":int,
        "fixed":list, "print-only-breakers":bool,"store-sym":str, "addsym":str, "asp":bool, "pb":int,
        "logfile":str
    }

    def parse_kwargs(self, kwargs):
        parsed = []
        for  key, value in kwargs.items():
            key = key.replace("_","-")
            if key not in self.supported_kwargs:
                raise ValueError(f"Unknown parameter: {key}")
            if not isinstance(value, self.supported_kwargs[key]):
                raise ValueError(f"Expected parameter {key} to be of type {self.supported_kwargs[key]} but got  {value} of type {type(value)}")

            if isinstance(value, bool):
                if value is True: # toggle
                    parsed += [f"-{key}"]
            elif isinstance(value, str):
                parsed += [f"-{key}", value]
            elif isinstance(value, int):
                parsed += [f"-{key}", str(value)]
            elif isinstance(value, list):
                parsed += [f"-{key}"] + [value]
            else:
                raise ValueError(f"Unknown type {type(value)} for value of parameter {key}")

        return parsed

    def run(self, input=None, output=None, stdout=PIPE, stderr=STDOUT, **kwargs):
        """
            Run BreakID tool with arguments
        """
        tmp = tempfile.NamedTemporaryFile().name
        with open(tmp, "w") as f:
            f.write(input)

        parsed = self.parse_kwargs(kwargs | {"f": tmp})
        res = subprocess.run([self.path_to_binary] + parsed, stdout=stdout, stderr=stderr)
        if res.returncode > 0:
            raise ValueError(f"Something went wrong, BreakID exitetd with non-zero exitcode ({res.returncode})\n\n" + res.stdout.decode("utf"))

        os.remove(tmp)
        return res.stdout.decode()

    def break_model(self, model, format="dimacs", **kwargs):
        """
            Find symmetries in CPMpy model and add breaking constraints to it.
        """
        if model.objective_ is not None:
            raise NotImplementedError("TODO: implement strong symmetry breaking with pb")

        breaking_cons = self.get_breakers(model.constraints, format=format, **kwargs)
        if format == "opb":
            breaking_cons, obj = breaking_cons
        else:
            obj = None

        model = cp.Model(model.constraints + breaking_cons)
        if obj is not None:
            model.minimize(obj)

        return model


    def get_input(self, constraints, format, subset=None):
        """
            Construct textual input from constraints, based on required format.
            :param: constraints: list of CPMpy constraints
            :param: format: the format to use, should be 'dimacs' or 'opb'
            :param: subset: if not none, generate partial symmetries that map variables in `subset` to `subset`.
                                uses "strong symmetry detection" in breakid so only supported when format is 'opb'
        """
        if format == "dimacs":
            if subset is not None: raise ValueError("Cannot find generators over subset of variables in DIMACS format")
            return self._to_dimacs(constraints)
        elif format == "opb":
            obj = None if subset is None else cp.sum(subset)
            return self._to_opb(constraints, obj=obj)
        else:
            raise ValueError(f"Unknown format {format}, chose from ['dimacs', 'opb']")

    def get_breakers(self, constraints, format="dimacs", subset=None, **kwargs):
        """
            Get symmetry breaking constraints
            :param: constraints: list of CPMpy constraints
            :param: format: the format to use, should be 'dimacs' or 'opb'
            :param: subset: if not none, generate partial symmetries that map variables in `subset` to `subset`.
                                uses "strong symmetry detection" in breakid so only supported when format is 'opb'
            :param: kwargs: additional keyword arguments to pass to the breakid tool
        """
        if format == "opb" and "pb" not in kwargs:
            kwargs['pb'] = 0

        input = self.get_input(constraints, format, subset)
        output = self.run(input=input, print_only_breakers=True, **kwargs)
        output = "\n".join(l for l in output.splitlines() if l[0] not in {"*", "c"})  # remove comments

        if format == "dimacs":
            return self._parse_dimacs(output)
        elif format == "opb":
            cpm_breakers, obj = self._parse_opb(output, sep="\n")
            assert obj is None
            return cpm_breakers

    def get_generators(self, constraints, format="dimacs", subset=None, symfile=None, **kwargs):
        """
            Run BreakdID and return generators detected.
            :param: constraints: list of CPMpy constraints
            :param: format: the format to use, should be 'dimacs' or 'opb'
            :param: subset: if not none, generate partial symmetries that map variables in `subset` to `subset`.
                                uses "strong symmetry detection" in breakid so only supported when format is 'opb'
            :param: symfile: optional: where to store the textual representation of the generators
            :param: kwargs: additional keyword arguments to pass to the breakid tool

            Returns a list of lists: (`Permutation+`, 'RowSymmetry+')
            Run BreakID and parse generators into `Symmetry` objects.
        """

        if format == "opb" and "pb" not in kwargs:
            kwargs['pb'] = 0

        input = self.get_input(constraints, format, subset)
        if subset is not None:
            subset = frozenset(subset)

        do_remove = False
        if symfile is None:
            do_remove = True
            symfile = tempfile.NamedTemporaryFile().name
        output = self.run(input=input, store_sym=symfile, **kwargs)
        with open(symfile, "r") as f:
            generators = f.read()

        lines = generators.splitlines()
        bvs = [k for k,v in natsorted(self._map.items(), key=lambda kv : kv[1])]

        i = 0
        cpm_permutations = []
        cpm_matrices = []
        while i < len(lines):
            line = lines[i]
            if line.startswith("("): # permutation
                cpm_group = []
                groups = line.strip()[1:-1].split(") (")
                for group in groups:
                    group = group.strip()
                    tup = []
                    for var_idx in map(int,group.split(" ")):
                        bv = bvs[abs(var_idx)-1]
                        lit = bv if var_idx > 0 else ~bv
                        if subset is not None and lit not in subset:
                            break
                        tup.append(lit)
                    else: # only add if all vars in subset
                        cpm_group.append(tuple(tup))
                if len(cpm_group):
                    cpm_permutations.append(Permutation(cpm_group))
                i += 1 # go to next line
            elif line.startswith("rows"):
                # parse header of matrix
                n_rows, n_cols = [int(x) for i,x in enumerate(line.split(" ")) if i % 2 == 1]
                matrix = [[None for c in range(n_cols)] for r in range(n_rows)] # init matrix
                for r in range(n_rows):
                    for c, var_idx in enumerate(map(int, lines[i+r+1].strip().split(" "))):
                        matrix[r][c] = bvs[abs(var_idx)-1] if var_idx > 0 else ~bvs[abs(var_idx)-1]

                # check which columns correspond to subset
                if subset is not None:
                    np_matrix = np.array(matrix)
                    col_idxes = []
                    for j, col in enumerate(np_matrix.T):
                        if any(v in subset for v in col):
                            assert all(v in subset for v in col), f"BreakID should only map vars in subset to other vars in subset, but got mapping {col}"
                            col_idxes.append(j)
                    matrix = np_matrix[:, col_idxes].tolist()

                cpm_matrices.append(RowSymmetry(matrix))
                i += n_rows+1

        if do_remove:
            os.remove(symfile)
        return cpm_permutations, cpm_matrices

    def breakers_from_generators(self, generators, format="opb", symfile=None, **kwargs):
        """
            Runs BreakID with empty theory and injects stored symmetries.
            Returns set of constraints, either clausal or pb
            :param: generators: list of generators (of class `Symmetry`)
            :param: format: the format to use, should be 'dimacs' or 'opb'
            :param: kwargs: additional keyword arguments to pass to the breakid tool
        """
        if format != "opb": raise NotImplementedError("TODO: implement parsing of cnf breakers from file")

        tmp = tempfile.NamedTemporaryFile().name
        vars = set().union(*[gen.get_variables() for gen in generators])
        self._map = {v : i+1 for i, v in enumerate(natsorted(vars, key=str))}

        with open(tmp, "w") as f:
            for gen in generators:
                if isinstance(gen, Permutation):
                    f.write(" ".join(["( " + " ".join(str(self._map[v]) for v in tup) + " )" for tup in gen.perm]) + "\n")
                elif isinstance(gen, RowSymmetry):
                    f.write("rows {} columns {}\n".format(*gen.matrix.shape))
                    for row in gen.matrix:
                        f.write(" ".join(str(self._map[v]) for v in row))
                        f.write("\n")
                else:
                    raise ValueError("Unknown symmetry type:", type(gen))

        # construct empty theory
        if format == "dimacs":
            input = f"p cnf {len(self._map)} 0"
        elif format == "opb":
            input = f"* #variable= {len(self._map)} #constraint= 0"
        else:
            raise ValueError("Unknown format:", format)

        output = self.run(input=input,addsym=tmp, **kwargs)

        # remove comments
        lines = output.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("* #variable="))
        output = "\n".join(lines[start+1:])

        os.remove(tmp)

        if format == "dimacs":
            return self._parse_dimacs(output)
        else:
            return self._parse_opb(output, sep="\n")[0] # no objective returned



    def _to_dimacs(self, constraints):
        """
            Convert list of CPMPy constraints to DIMACS format
            Uses CPMpy's internal transformations and hence, the output of this function may change depending on
                the version of CPMpy used.
        """
        constraints = to_cnf(flatten_constraint(constraints))
        self._map = {v : i+1 for i, v in enumerate(natsorted(get_variables(constraints), key=str))}

        out = f"p cnf {len(self._map)} {len(constraints)}\n"
        for cons in constraints:

            if isinstance(cons, _BoolVarImpl):
                cons = Operator("or", [cons])

            if isinstance(cons, Operator) and cons.name == "->":
                # implied constraint
                cond, subexpr = cons.args
                assert isinstance(cond, _BoolVarImpl)

                # implied boolean variable, convert to unit clause
                if isinstance(subexpr, _BoolVarImpl):
                    subexpr = Operator("or", [subexpr])

                # implied clause, convert to clause
                if isinstance(subexpr, Operator) and subexpr.name == "or":
                    cons = Operator("or", [~cond] + subexpr.args)
                else:
                    raise ValueError(f"Unknown format for CNF-constraint: {cons}")

            if isinstance(cons, Comparison):
                raise NotImplementedError(f"Pseudo-boolean constraints not (yet) supported!")

            assert isinstance(cons, Operator) and cons.name == "or", f"Should get a clause here, but got {cons}"

            # write clause to cnf format
            ints = []
            for v in cons.args:
                if isinstance(v, NegBoolView):
                    ints.append(str(-self._map[v._bv]))
                elif isinstance(v, _BoolVarImpl):
                    ints.append(str(self._map[v]))
                else:
                    raise ValueError(f"Expected Boolean variable in clause, but got {v} which is of type {type(v)}")

            out += " ".join(ints + ["0"]) + "\n"
        return out[:-1]

    def _parse_dimacs(self, string, sep=" "):

        # get map of variables to indices
        bvs = [k for k,v in natsorted(self._map.items(), key=lambda kv : kv[1])]
        lines = string.splitlines()
        header = lines[0].strip()

        if not header.startswith("p"):
            # not a header in the file, skip
            start = 0
        else:
            start = 1

        constraints = []
        for line in lines[start:]:

            if line is None or len(line) <= 0:
                break

            str_idxes = line.strip().split(sep)
            clause = []
            for i, var_idx in enumerate(map(int, str_idxes)):
                if abs(var_idx) > len(bvs):  # var does not exist yet, create
                    bvs += [cp.boolvar() for _ in range(abs(var_idx) - len(bvs))]

                if var_idx > 0:  # boolvar
                    clause.append(bvs[var_idx - 1])
                elif var_idx < 0:  # neg boolvar
                    clause.append(~bvs[(-var_idx) - 1])
                elif var_idx == 0:  # end of clause
                    assert i == len(
                        str_idxes) - 1, f"Can only have '0' at end of a clause, but got 0 at index {i} in clause {str_idxes}"
            constraints.append(cp.any(clause))

        return constraints

    def _to_opb(self, constraints, obj=None):
        """
            Write any model containing only Boolean constraints to .opb-formatted string.
            Uses CPMpy transformation pipeline (similar to Exact's) to linearize all constraints.
            Hence, the output of this function may change depending on the version of CPMpy used.
        """
        constraints = toplevel_list(constraints)
        constraints = decompose_in_tree(constraints,supported=frozenset({'alldifferent'}))  # Alldiff has a specialzed MIP decomp
        constraints = flatten_constraint(constraints)  # flat normal form
        constraints = reify_rewrite(constraints, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        constraints = only_numexpr_equality(constraints, supported=frozenset(["sum", "wsum"]))  # supports >, <, !=
        constraints = only_bv_reifies(constraints)
        constraints = only_implies(constraints)  # anything that can create full reif should go above...
        constraints = linearize_constraint(constraints, supported=frozenset({"sum", "wsum"}))  # the core of the MIP-linearization
        constraints = only_positive_bv(constraints)

        def format_comparison(cons):

            assert isinstance(cons, Comparison), f"Expected comparison, but got {cons}"
            lhs, rhs = cons.args

            assert isinstance(lhs, Operator) and lhs.name == "wsum", f"Expected weighted sum here, but got {lhs}"
            assert is_num(rhs), f"Should be numerical rhs of comparison, but got {rhs} of type {type(rhs)}"
            for w, v in zip(*lhs.args):
                if not v.is_bool(): raise ValueError(f"Expected all Boolean variables in lhs, but got {v} in {lhs}")

            # linear constraints can be <=,== or >=
            if cons.name == "==":
                return format_comparison(lhs <= rhs) + format_comparison(lhs >= rhs)
            if cons.name == "<=":
                lhs = cp.sum(-w * v for w,v in zip(*lhs.args))
                rhs = -rhs
                cons = lhs >= rhs

            assert cons.name == ">="
            return [format_wsum(lhs) + f" >= {rhs};"]

        def format_wsum(wsum_expr):
            assert isinstance(wsum_expr, Operator) and wsum_expr.name == "wsum"

            string = ""
            for w,v in zip(*wsum_expr.args):
                string += " +" if w > 0 else " -"
                string += f"{abs(w)} x{self._map[v]}"
            return string[1:] # remove leading space

        def _to_wsum(expr):
            assert not expr.is_bool(), f"Expected numerical expression here, but got {expr}"
            if isinstance(expr, _NumVarImpl):
                expr = Operator("wsum", [[1], [expr]])
            elif isinstance(expr, Operator) and expr.name == "sum":
                expr = Operator("wsum", [[1] * len(expr.args), expr.args])
            assert isinstance(expr, Operator) and expr.name == "wsum", f"Expected weighted sum here, but got {expr}"
            return expr

        def bigM_le(cond, subexpr):
            lhs, rhs = subexpr.args
            assert isinstance(lhs, Operator) and lhs.name == "wsum"

            M = rhs - get_bounds(lhs)[1]  # max bound of lhs
            lhs.args[0].insert(0, M)
            lhs.args[1].insert(0, ~cond)
            return format_comparison(only_positive_bv([lhs <= rhs])[0])

        def bigM_ge(cond, subexpr):
            lhs, rhs = subexpr.args
            assert isinstance(lhs, Operator) and lhs.name == "wsum"

            M = rhs - get_bounds(lhs)[0]  # min bound of lhs
            lhs.args[0].insert(0, M)
            lhs.args[1].insert(0, ~cond)
            return format_comparison(only_positive_bv([lhs >= rhs])[0])

        str_cons = []
        flipmap = {"<=":">=", "==":"==", ">=":"<="}

        # keep map of variables to variable indices, used during parsing too
        self._map = {v : i+1 for i, v in enumerate(natsorted(get_variables(constraints), key=str))}

        for cons in constraints:
            """Constraints are weighted linear sums, or half-reification thereof"""
            if isinstance(cons, Comparison):
                lhs, rhs = cons.args
                lhs = _to_wsum(lhs) # can also be numvar still
                str_cons += format_comparison(Comparison(cons.name, lhs,rhs))

            elif isinstance(cons, Operator) and cons.name == "->":
                cond, subexpr = cons.args
                assert isinstance(cond, _BoolVarImpl)
                assert isinstance(subexpr, Comparison)
                assert is_num(subexpr.args[1])
                subexpr.args[0] = _to_wsum(subexpr.args[0])
                if subexpr.args[1] <= 0:
                    subexpr.args[0].args[0] = [-w for w in subexpr.args[0].args[0]]
                    subexpr.args[1] = - subexpr.args[1]
                    subexpr.name = flipmap[subexpr.name]


                if subexpr.name == "<=":
                    str_cons += bigM_le(cond, subexpr)
                elif subexpr.name == ">=":
                    str_cons += bigM_ge(cond, subexpr)
                elif subexpr.name == "==":
                    str_cons += bigM_le(cond, subexpr)
                    str_cons += bigM_ge(cond, subexpr)
                else:
                    raise ValueError(f"Unexpected comparison in reification: {cons}")
            else:
                raise ValueError(f"Expected linear sum or reification thereof, but got {cons}")

        if obj is not None:
           obj = "min: " + format_wsum(_to_wsum(obj)) + ";\n"
        else:
            obj = ""

        header = f"* #variable= {len(self._map)} #constraint= {len(str_cons)}\n"
        return header + obj + "\n".join(str_cons)

    def _parse_opb(self, string, sep=";"):
        """
            Parse pseudo-boolean model (.opb formatted)
        """
        lines = string.split("\n")
        remainder = "\n".join(l for l in lines if not l.startswith("*"))
        constraints = remainder.split(sep)
        bvs = [k for k,v in natsorted(self._map.items(), key=lambda kv : kv[1])]

        def parse_wsum(string): # parse weighted sum
            string = string.strip()
            splitted = [x for x in string.split(" ") if len(x)]
            weights, vars = [], []
            for i,x in enumerate(splitted):
                if x == "-":
                    splitted[i+1] = -splitted[i+1]
                elif x == "+":
                    pass
                elif x.isnumeric() or x.startswith("-") or x.startswith("+"):
                    weights.append(int(x))
                    # found weight
                elif x.startswith("x"):
                    # found variable
                    var_idx = int(x[1:])
                    if var_idx > len(bvs):  # var does not exist yet, create
                        bvs.extend([cp.boolvar() for _ in range(var_idx - len(bvs))])
                    vars.append(bvs[var_idx-1])
                else:
                    raise ValueError(f"Unexpected string {x}; found while parsing {string}")
            return Operator("wsum", [weights, vars])

        cpm_cons = []
        obj = None
        for cons in constraints:
            if cons.startswith("min"): # objective
                obj = parse_wsum(cons.split("min:")[-1])
            else: # comparison constraint
                if ">=" in cons:
                    lhs, rhs = cons.split(">=")
                else:
                    raise ValueError(f"Unexpected comparison consrtaint, BreakID should return normalized constraints, but got {cons}")

                lhs = parse_wsum(lhs)
                rhs = int(rhs.strip())
                cpm_cons.append(lhs >= rhs)

        return cpm_cons, obj