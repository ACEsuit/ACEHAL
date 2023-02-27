import pytest

import glob
from pathlib import Path

import numpy as np

import ase.io
from ase.calculators.emt import EMT
from ase.atoms import Atoms

from ACEHAL.HAL import HAL
from sklearn.linear_model import BayesianRidge
from ACEHAL.optimize_basis import basis_dependency_range_max


def test_T_P_ramps_and_per_config_params(fit_data, monkeypatch, tmp_path):
    # test as many variants as possible in 1 run, since test is slow

    fixed_basis_info = {"r_cut": 5.0, "smoothness_prior": None}

    optimize_params = {"cor_order": ("int", (2, 3)), "maxdeg": ("int", (4, 8))}
    basis_dependency_source_target = ("cor_order", "maxdeg")
    do_HAL_test(None, fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path,
                T_K=[(100, 1000)], P_GPa=[(0.25, 0.5), (0.5, 1.0)])

    # make sure temperature ramped up
    traj = []
    for f in glob.glob("test_HAL.traj.it_*.extxyz"):
        ats = ase.io.read(f, ":")
        if len(ats) > len(traj):
            traj = ats
    # at least 500 steps
    assert len(traj) > 50
    # check T ramp for last 250 to 100..350
    assert np.mean([at.get_kinetic_energy() for at in traj[-15:]]) / np.mean([at.get_kinetic_energy() for at in traj[5:5+15]]) > 3.0
    # check that cell changed
    assert not np.all(traj[0].cell == traj[-1].cell)
    # check something about tau_rel ramp?


def do_HAL_test(basis_source, fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path, T_K=1000.0, P_GPa=None):
    monkeypatch.chdir(tmp_path)

    np.random.seed(10)

    fit_configs, test_configs, E0s, data_keys, weights = fit_data

    fixed_basis_info["elements"] = list(E0s.keys())
    # filter allowed ranges to avoid exceeding max len
    basis_dependency_range_max({"julia_source": basis_source}, fixed_basis_info, optimize_params, 200,
                               basis_dependency_source_target[0], basis_dependency_source_target[1])

    solver = BayesianRidge(fit_intercept=False, compute_score=True)

    print("calling HAL with range limited optimize_params", optimize_params)

    n_iters = 10

    # copy per-config params
    starting_configs = [at.copy() for at in fit_configs]
    for at_i, at in enumerate(starting_configs):
        if isinstance(T_K, list):
            at.info["HAL_traj_params"] = at.info.get("HAL_traj_params", {})
            at.info["HAL_traj_params"]["T_K"] = T_K[at_i % len(T_K)]
        if isinstance(P_GPa, list):
            at.info["HAL_traj_params"] = at.info.get("HAL_traj_params", {})
            at.info["HAL_traj_params"]["P_GPa"] = P_GPa[at_i % len(P_GPa)]
    # back to boring default values if per-config list was passed in
    if isinstance(T_K, list):
        T_K = 1000.0
    if isinstance(P_GPa, list):
        P_GPa = None

    new_fit_configs, basis_info, new_test_configs = HAL(
            fit_configs, starting_configs, basis_source, solver,
            fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights},
            n_iters=n_iters, ref_calc=EMT(),
            traj_len=1000, dt=1.0, tol=0.3, tau_rel=0.5, T_K=T_K, P_GPa=P_GPa,
            basis_optim_kwargs={"n_trials": 20,
                                "max_basis_len": 200,
                                "fixed_basis_info": fixed_basis_info,
                                "optimize_params": optimize_params,
                                "seed": 5},
            basis_optim_interval=5, file_root="test_HAL",
            test_fraction=0.3)

    assert len(new_fit_configs) < n_iters
    assert len(new_test_configs) > 0

    # make sure the right number of files are present
    # matching numbers of configs
    assert len(list(tmp_path.glob("test_HAL.config_fit.it_*.extxyz"))) == len(new_fit_configs)
    assert len(list(tmp_path.glob("test_HAL.config_test.it_*.extxyz"))) == len(new_test_configs)
    # can't predict exact # of potentials - at least initial one and one per new config, but perhaps also more after
    # basis optimizations
    assert len(list(tmp_path.glob("test_HAL.pot.it_*.json"))) >= len(new_fit_configs) + 1
    # one dimer plot for each fit
    assert len(list(tmp_path.glob("test_HAL.dimers.it_*.pdf"))) == len(list(tmp_path.glob("test_HAL.pot.it_*.json")))
    # trajectory and plot for each iter
    assert len(list(tmp_path.glob("test_HAL.traj.it_*.extxyz"))) == n_iters
    assert len(list(tmp_path.glob("test_HAL.run_data.it_*.pdf"))) == n_iters

    for f in tmp_path.glob("test_HAL.config_*.it_*.extxyz"):
        assert len(ase.io.read(f, ":")) == 1

    for f in tmp_path.glob("test_HAL.traj.it_*.extxyz"):
        l = len(ase.io.read(f, ":"))
        assert l > 1 and l <= 101
        print("do_HAL_test got len(traj)", l)
