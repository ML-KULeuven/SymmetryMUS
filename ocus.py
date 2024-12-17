
import cpmpy as cp
from cpmpy.tools.explain import make_assump_model

from breakid import BreakID, BREAKID_PATH


def ocus(soft, hard=[], solver="ortools", hs_solver="ortools", lex_leaders=True, generate_symmetric=True):

    model, soft, assump = make_assump_model(soft,hard)

    s = cp.SolverLookup.get(solver, model)
    hs_solver = cp.SolverLookup.get(hs_solver)
    hs_solver.minimize(cp.sum(assump))

    sets_to_hit = []

    if lex_leaders is True or generate_symmetric is True:
        breakid = BreakID(BREAKID_PATH)
        perms, matrices = breakid.get_generators(model.constraints, format="opb", pb=31, subset=assump)
        symmetries = perms + matrices
    if lex_leaders is True:
        hs_solver += breakid.breakers_from_generators(symmetries, format="opb", pb=31)


    while hs_solver.solve() is True:

        hitting_set = [a for a in assump if a.value()]

        if s.solve(assumptions=hitting_set) is False:
            break # UNSAT, found MUS
        # else, SAT, need to grow
        sat_subset = [a for a,c in zip(assump, soft) if a.value() or c.value()]
        while s.solve(assumptions=sat_subset) is True:
            corr_subset = [a for a,c in zip(assump, soft) if not a.value() and not c.value()]
            if generate_symmetric is True: # also find symmetric versions of `subset`
                sat_subset = set(sat_subset)
                for sym in symmetries:
                    new_corr_subsets = sym.apply_symmetry(corr_subset, exclude=frozenset(sets_to_hit))
                    for mcs in new_corr_subsets:
                        hs_solver += cp.any(mcs)  # add mcs to hitting-set solver
                        sets_to_hit.append(mcs)   # keep track of sets to hit
                        sat_subset |= set(mcs)    # extend sat_subset
                sat_subset = list(sat_subset)
            else:
                hs_solver += cp.any(corr_subset)
                sets_to_hit.append(corr_subset)
                sat_subset.extend(corr_subset)

    return [c for a,c in zip(assump, soft) if a in frozenset(hitting_set)]

