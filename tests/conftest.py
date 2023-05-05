# make sure this happens when pytest starts, before _anything_ is imported and starts a julia
# workspace
import os
from pathlib import Path
os.environ["JULIA_PROJECT"] = str(Path(__file__).parent / "julia_assets")

import pytest

import numpy as np

from ase.atoms import Atoms
from ase.calculators.emt import EMT

from ACEHAL import basis, fit

from sklearn.linear_model import BayesianRidge

# fixtures that fit a model

@pytest.fixture(scope="module")
def fit_data():
    calc = EMT()
    E0s = {}
    for sym in ['Al', 'Cu']:
        at_isolated = Atoms(sym)
        at_isolated.calc = calc
        E0s[sym] = at_isolated.get_potential_energy()

    rng = np.random.default_rng(5)
    at_prim = Atoms('AlAlAlCu', positions=[[0, 0, 0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0]],
                    cell=[4.0] * 3, pbc=[True])

    fit_configs = []
    for _ in range(10):
        sc = at_prim * (3, 3, 3)
        sc.rattle(0.01, rng=rng)
        F = np.eye(3) + rng.normal(scale=0.02, size=(3,3))
        sc.set_cell(sc.cell @ F, True)
        sc.calc = calc
        sc.info["REF_energy"] = sc.get_potential_energy()
        sc.arrays["REF_forces"] = sc.get_forces()
        sc.info["REF_virial"] = -sc.get_volume() * sc.get_stress(voigt=False)
        sc.calc = None
        fit_configs.append(sc)

    test_configs = []
    for _ in range(10):
        sc = at_prim * (3, 3, 3)
        sc.rattle(0.01, rng=rng)
        F = np.eye(3) + rng.normal(scale=0.02, size=(3,3))
        sc.set_cell(sc.cell @ F, True)
        sc.calc = calc
        sc.info["REF_energy"] = sc.get_potential_energy()
        sc.arrays["REF_forces"] = sc.get_forces()
        sc.info["REF_virial"] = -sc.get_volume() * sc.get_stress(voigt=False)
        sc.calc = None
        test_configs.append(sc)

    # fix weights to be per atom
    for at in fit_configs + test_configs:
        at.info["REF_energy_weight"] = 1.0 / len(at)
        at.info["REF_virial_weight"] = 1.0 / len(at)

    data_keys = {'E': 'REF_energy', 'F': 'REF_forces', 'V': 'REF_virial'}
    weights = {'E_per_atom' : 10.0, 'F': 1.0, 'V_per_sqrt_atom': 1.0}

    ## less data, worse transferability
    fit_configs = fit_configs[0:3]

    return fit_configs, test_configs, E0s, data_keys, weights


@pytest.fixture(scope="module")
def fit_model_all_info(fit_data):
    fit_configs, _, E0s, data_keys, weights = fit_data

    basis_info = {'elements': list(E0s), 'cor_order': 2, 'maxdeg': 6,
                  'r_cut': 3.75, 'smoothness_prior': None}
    B_len_norm = basis.define_basis(basis_info)

    solver = BayesianRidge(fit_intercept=False, compute_score=True)

    n_observations = len(fit_configs) * (1 + 6) + 3 * np.sum([len(at) for at in fit_configs])

    # make deterministic
    rng = np.random.default_rng(seed=10)
    calc, Psi, Y, coef, prop_row_inds = fit.fit(fit_configs, solver, B_len_norm, E0s,
                                     data_keys=data_keys, weights=weights,
                                     n_committee=16, rng=rng,
                                     return_linear_problem=True)

    return calc, Psi, Y, coef, prop_row_inds, n_observations, B_len_norm


@pytest.fixture(scope="module")
def fit_model(fit_model_all_info):
    return fit_model_all_info[0]
