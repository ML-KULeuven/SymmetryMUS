
from breakid import BreakID
from breakid import BREAKID_PATH

import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model


def deletion_based_mus(soft, hard=[], solver="exact", use_symmetries=True):
    """
        Lightweight CPMPy implementation of the deletion-based MUS algorithm.
        Optimizations implemented:
            - Incremental solving using assumption variables
            - Clause set refinement [1]
            - Optional: removing symmetric transition constraints
            - TODO: model rotation?
        """

    model, soft, assump = make_assump_model(soft, hard)

    if use_symmetries:
        breakid = BreakID(BREAKID_PATH)  # use pb branch
        permutations, matrices = breakid.get_generators(model.constraints, format="opb", subset=assump,pb=31)
        symmetries = permutations + matrices

    s = cp.SolverLookup.get(solver, model)
    assert s.solve(assump=assump) is False, "Model should be UNSAT in order to find a MUS"

    to_check = set(s.get_core())
    core = set(s.get_core)
    while len(to_check):

        test_var = to_check.pop()

        subassump = list(core - {test_var})
        if s.solve(subassump=subassump) is False:
            # still unsat, reduce with potential smaller core
            assert set(s.get_core()) <= set(subassump), "Solver core should be subset of current core"
            core = set(s.get_core())
        else:
            # SAT, test_var is a transition constraint
            if use_symmetries: # find symmetric transition constraints
                for sym in symmetries:
                    can_skip = sym.get_symmetric_images_in_subset(core, test_var)
                    to_check -= can_skip # remove symmetric transition constraints

    return [s for a, s in zip(assump, soft) if a in frozenset(core)]