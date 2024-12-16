
BREAKID_PATH = "/local/usr/bin/breakid/src/BreakID"


def unroll(subsets, symmetries):

    all_subsets = []
    subsets = list(subsets)
    while len(subsets):
        ss = subsets.pop(0)
        all_subsets.append(ss)

        for sym in symmetries: # find all symmetric images
            sym_ss_generator = sym.apply_symmetry(subset=ss, exclude=frozenset(all_subsets + subsets))
            subsets += list(sym_ss_generator)

    return all_subsets

