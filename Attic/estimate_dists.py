from scipy.signal import argrelextrema

import ase.data
try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except ModuleNotFoundError:
    from ase.neighborlist import neighbor_list

def estimate_dists_per_pair(atoms_list, min_cutoff=2.0, bin_width=0.1):
    """Estimate r_in and r_0 from list of atomic configurations

    `r_in_sym` is defined as the shortest distance for each species pair indexed by their
    chemical symbols.  `r_0_sym` is defined as the position of the first local maximum in 
    the neighbor list histogram for each species pair.

    Parameters
    ----------
    atoms_list: list(Atoms)
        list of atomic configurations
    min_cutoff: float, default 1.0
        minimum neighbor cutoff to check
    bin_width: float, default 0.1
        approximate width of bins in histogram used to find typical distance r_0

    Returns
    -------
    r_in_sym dict((str, str): float) dict of innermost distance found for each species pair
    r_0_sym dict((str, str): float) dict of typical distance found for each species pair

    """
    Zs = set()
    for atoms in atoms_list:
        Zs |= set(atoms.numbers)
    sym_pairs = [(ase.data.chemical_symbols[Z0], ase.data.chemical_symbols[Z1]) for Z0 in Zs for Z1 in Zs if Z0 <= Z1]

    r_0_sym = {'dummy': None}
    cutoff = min_cutoff
    while any([r_0 is None for r_0 in r_0_sym.values()]):
        # gather dists by species pair
        dists = {sym_pair: [] for sym_pair in sym_pairs}
        for atoms in atoms_list:
            ii, jj, dd = neighbor_list('ijd', atoms, cutoff)
            for sym_pair in sym_pairs:
                dists[sym_pair].extend(dd[np.logical_and(atoms.symbols[ii] == sym_pair[0], atoms.symbols[jj] == sym_pair[1])])

        # find distances for each pair
        r_0_sym = {}
        r_in_sym = {}
        for sym_pair in sym_pairs:
            if len(dists[sym_pair]) == 0:
                # no neighbors for this cutoff, mark and skip to next cutoff
                r_in_sym[sym_pair] = None
                r_0_sym[sym_pair] = None
                break

            # r_in is explicit minimum
            r_in = np.min(dists[sym_pair])

            # r_0 from first max of histogram
            n_bins = max(int(np.round((max(dists[sym_pair]) - min(dists[sym_pair])) / bin_width)), 1)
            nums, bins = np.histogram(dists[sym_pair], bins=n_bins)
            bin_of_max_list = argrelextrema(nums, np.greater)[0]
            if len(bin_of_max_list) > 0:
                bin_of_max = bin_of_max_list[0]
                r_0 = 0.5 * (bins[bin_of_max] + bins[bin_of_max + 1])
            else:
                # no maximum yet
                r_0 = None

            r_in_sym[sym_pair] = r_in
            r_0_sym[sym_pair] = r_0

        cutoff *= 1.5

    return r_in_sym, r_0_sym


