import pytest

import numpy as np

from copy import deepcopy

from sklearn.linear_model import BayesianRidge

from ase.atoms import Atoms
from ACEHAL.optimize_basis import estimate_dists_per_pair, basis_dependency_range_max, optimize

def test_estimate_dists():
    np.random.seed(10)

    atoms_list = []
    for _ in range(100):
        atoms = Atoms(symbols='AlAlCuCu', positions=[
                                          [0   + 0.1 * np.random.normal(), 0.0, 0.0],
                                          [2.0 + 0.1 * np.random.normal(), 0.0, 0.0],
                                          [3.0 + 0.1 * np.random.normal(), 0.0, 0.0],
                                          [5.0 + 0.1 * np.random.normal(), 0.0, 0.0]],
                      cell=[12] * 3, pbc=[True] * 3)
        atoms_list.append(atoms)

    r_in_Z, r_0_Z = estimate_dists_per_pair(atoms_list)
    print("r_in", r_in_Z, "r_0", r_0_Z)

    assert r_in_Z[('Cu', 'Cu')] == pytest.approx(1.61, abs=0.01)
    assert r_in_Z[('Al', 'Cu')] == pytest.approx(0.64, abs=0.01)
    assert r_in_Z[('Al', 'Al')] == pytest.approx(1.66, abs=0.01)

    assert r_0_Z[('Cu', 'Cu')] == pytest.approx(1.97, abs=0.01)
    assert r_0_Z[('Al', 'Cu')] == pytest.approx(0.89, abs=0.01)
    assert r_0_Z[('Al', 'Al')] == pytest.approx(1.92, abs=0.01)


# might be good to test the functionality that limits dependency source range if any of it
# has no possible target range, but hard to get there with a single component database 
def test_basis_dep_range(fit_data):
    _, _, E0s, _, _ = fit_data

    fixed_basis_info = {"elements": list(E0s.keys()), "r_cut": 4.5, "r_in": 2.0, "pairs_r_dict": {}}

    optimize_params = {"r_0": ("float", (4.0, 5.0)), "cor_order": ("int", (2, 8)),
                       "maxdeg": ("int", (8, 18))}

    basis_dependency_range_max({}, fixed_basis_info, optimize_params, 100, "cor_order", "maxdeg")

    optimize_params_range = deepcopy(optimize_params)
    optimize_params_range["cor_order"] = ("int", (2, 4))
    optimize_params_range["maxdeg"] = ("int", ("cor_order", {2: (8, 12), 3: (8, 9),
                                                             4: (8, 8)}))

    assert optimize_params_range == optimize_params


def test_optimize_basis(fit_data):
    fit_configs, test_configs, E0s, data_keys, weights = fit_data

    fixed_basis_info = {"elements": list(E0s.keys()), "r_cut": 4.5, "r_in": 2.0, "pairs_r_dict": {}}

    solver = BayesianRidge(fit_intercept=False, compute_score=True)
    basis_info = optimize(solver, fit_configs, 30,
             {"cor_order": ("int", (2, 3)),
              "maxdeg": ("int", ("cor_order", {2: (3, 5), 3: (3, 4)})),
              "r_0": ("float", (2.4, 3.2))},
             basis_kwargs={},
             fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights},
             fixed_basis_info=fixed_basis_info, max_basis_len=400, seed=5)

    # from known good run
    expected = {'cor_order': 2, 'maxdeg': 5, 'r_0': 3.1157737831026995}
    # fixed
    expected.update(fixed_basis_info)

    for k in list(basis_info.keys()):
        if k == "r_0":
            tol = 0.2
        else:
            tol = None
        assert basis_info.pop(k) == pytest.approx(expected[k], tol)
    assert len(basis_info) == 0


