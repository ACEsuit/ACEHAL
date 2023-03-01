import pytest

import numpy as np

from copy import deepcopy

from sklearn.linear_model import BayesianRidge

from ase.atoms import Atoms
from ACEHAL.optimize_basis import basis_dependency_range_max, optimize

# might be good to test the functionality that limits dependency source range if any of it
# has no possible target range, but hard to get there with a single component database 
def test_basis_dep_range(fit_data):
    _, _, E0s, _, _ = fit_data

    fixed_basis_info = {"elements": list(E0s.keys()), "r_cut": 4.5, "smoothness_prior": None}

    optimize_params = {"cor_order": ("int", (2, 5)), "maxdeg": ("int", (4, 18))}

    basis_dependency_range_max({}, fixed_basis_info, optimize_params, 200, "cor_order", "maxdeg")
    print("BOB got optimize_params", optimize_params)

    optimize_params_range = deepcopy(optimize_params)
    optimize_params_range["cor_order"] = ("int", (2, 5))
    optimize_params_range["maxdeg"] = ("int", ("cor_order", {2: (4, 7), 3: (4, 5), 4: (4, 4), 5: (4,4)}))

    assert optimize_params_range == optimize_params


def test_optimize_basis(fit_data):
    fit_configs, test_configs, E0s, data_keys, weights = fit_data

    fixed_basis_info = {"elements": list(E0s.keys()), "r_cut": 4.5, "smoothness_prior": None}

    solver = BayesianRidge(fit_intercept=False, compute_score=True)
    basis_info = optimize(solver, fit_configs, 30,
             {"cor_order": ("int", (2, 3)),
              "maxdeg": ("int", ("cor_order", {2: (3, 5), 3: (3, 4)}))},
             basis_kwargs={},
             fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights},
             fixed_basis_info=fixed_basis_info, max_basis_len=400, seed=5)

    # from known good run
    expected = {'cor_order': 2, 'maxdeg': 5}
    # fixed
    expected.update(fixed_basis_info)

    assert basis_info == expected
