
import cpmpy as cp
from cpmpy.tools.explain.utils import make_assump_model
from fontTools.misc.psOperators import PSOperators

from breakid import BreakID, BREAKID_PATH
from utils import unroll


def marco(soft, hard=[], solver="ortools", map_solver="ortools", use_symmetries=True, use_symmetries_in_shrink=False):

    model, soft, assump = make_assump_model(soft, hard)

    s = cp.SolverLookup.get(solver, model)
    assert s.solve(assumptions=assump) is False, "Model should be UNSAT in order to find a MUS"

    map_solver = cp.SolverLookup.get(map_solver)
    map_solver += cp.sum(assump) >= 0
    map_solver.solution_hint(assump, [1]*len(assump)) # preference for larger subsets

    if use_symmetries:
        breakid = BreakID(BREAKID_PATH)  # use pb branch
        permutations, matrices = breakid.get_generators(model.constraints, format="opb", subset=assump, pb=31)
        symmetries = permutations + matrices
        lexleaders = breakid.breakers_from_generators(symmetries, format="opb", pb=31)
        map_solver += lexleaders

    def _grow_to_mss(seed):
        candidates = set(assump) - set(seed)
        mss = list(seed)
        for a in candidates:
            if s.solve(assumptions=mss + [a]) is True:
                mss.append(a)
            else: # UNSAT, discard
                pass
        return mss

    def _shrink_to_mus(seed):
        to_check = set(seed)
        core = set(seed)
        while len(to_check):
            test_var = to_check.pop()
            subassump = list(core - {test_var})
            if s.solve(assumptions=subassump) is False:
                # still unsat, reduce with potential smaller core
                assert set(s.get_core()) <= set(subassump), "Solver core should be subset of current core"
                core = set(s.get_core())
            elif use_symmetries_in_shrink:  # find symmetric transition constraints (Disabled for AAAI25 paper results)
                for sym in symmetries:
                    can_skip = sym.get_symmetric_images_in_subset(core, test_var)
                    to_check -= can_skip  # remove symmetric transition constraints
            else: pass

        return list(core)

    musses = []
    # The MARCO algorithm
    while map_solver.solve():

        seed = [a for a in assump if a.value()]

        if s.solve(assumptions=seed): # SAT, grow to MSS
            mss = _grow_to_mss(seed)
            mcs = [a for a in assump if a not in frozenset(mss)]
            map_solver += cp.any(mcs) # ensure next seed hits this MCS

        else: # UNSAT, shrink to MUS, re-use deletion-based MUS code
            mus = _shrink_to_mus(seed)
            map_solver += ~cp.all(mus) # exclude super-sets of MUS
            musses.append(mus)

    print(f"Found {len(musses)} musses")
    if use_symmetries:
        # need to unroll to the full set of MUSes
        musses = unroll(musses, symmetries)

    return [[c for a,c in zip(assump, soft) if a in frozenset(mus)] for mus in musses]




