import pytest

import glob
from pathlib import Path

import numpy as np

import ase.io
from ase.calculators.emt import EMT
from ase.atoms import Atoms

from ACEHAL.HAL import HAL, _estimate_dists
from sklearn.linear_model import BayesianRidge
from ACEHAL.optimize_basis import basis_dependency_range_max


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

    basis_optim_kwargs = {}
    _estimate_dists(atoms_list, basis_optim_kwargs, "min")

    r_pairs_expected =  {('Cu', 'Cu'): {'r_in': 1.6077902819998728, 'r_0': 1.9721091138963613},
                         ('Al', 'Cu'): {'r_in': 0.639900037104955, 'r_0': 0.8856467607196443},
                         ('Al', 'Al'): {'r_in': 1.662435385860622, 'r_0': 1.9235919709831029}}
    r_elems_expected = {'Cu': {'r_in': 0.639900037104955, 'r_0': 0.8856467607196443},
                        'Al': {'r_in': 0.639900037104955, 'r_0': 0.8856467607196443}}

    r_in_global_expected = 0.639900037104955
    r_0_global_expected = 0.8856467607196443

    r_pairs = basis_optim_kwargs["fixed_basis_info"]["pairs_r_dict"]
    r_elems = basis_optim_kwargs["fixed_basis_info"]["elems_r_dict"]
    assert set(r_pairs.keys()) == set(r_pairs_expected.keys())
    for sym_pair in r_pairs:
        for d in ["r_in", "r_0"]:
            assert r_pairs[sym_pair][d] == pytest.approx(r_pairs_expected[sym_pair][d])

    assert set(r_elems.keys()) == set(r_elems_expected.keys())
    for sym in r_elems:
        for d in ["r_in", "r_0"]:
            assert r_elems[sym][d] == pytest.approx(r_elems_expected[sym][d])

    assert r_in_global_expected == pytest.approx(basis_optim_kwargs["fixed_basis_info"]["r_in"])
    assert r_0_global_expected == pytest.approx(basis_optim_kwargs["fixed_basis_info"]["r_0"])


def test_HAL_basis_default(fit_data, monkeypatch, tmp_path):
    fixed_basis_info = {"r_cut": 5.0, "r_0": 3.0, "r_in": 2.0, "pairs_r_dict": {}}

    optimize_params = {"cor_order": ("int", (2, 3)), "maxdeg": ("int", (4, 8))}
    basis_dependency_source_target = ("cor_order", "maxdeg")
    do_HAL_test(None, fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path)


@pytest.mark.skip(reason="basis too good, trajectory too slow")
def test_HAL_basis_smooth(fit_data, monkeypatch, tmp_path):
    fixed_basis_info = {"r_cut_ACE": 5.0, "r_cut_pair": 5.0, "r_0": 3.0, "agnesi_q": 4}

    optimize_params = {"cor_order": ("int", (2, 3)), "maxdeg_ACE": ("int", (4, 8)), "maxdeg_pair": ("int", (4, 8))}
    basis_dependency_source_target = ("cor_order", "maxdeg_ACE")
    do_HAL_test("ACEHAL.bases.smooth", fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path)


def test_cell_mc(fit_data, monkeypatch, tmp_path):
    fixed_basis_info = {"r_cut": 5.0, "r_0": 3.0, "r_in": 2.0, "pairs_r_dict": {}}

    optimize_params = {"cor_order": ("int", (2, 3)), "maxdeg": ("int", (4, 8))}
    basis_dependency_source_target = ("cor_order", "maxdeg")
    do_HAL_test(None, fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path, P_GPa=(0.0, 1.0))

    # make sure cell changed at least a little
    traj = ase.io.read("test_HAL.traj.it_09.extxyz", ":")
    assert not np.all(traj[0].cell == traj[-1].cell)


def test_T_ramp(fit_data, monkeypatch, tmp_path):
    fixed_basis_info = {"r_cut": 5.0, "r_0": 3.0, "r_in": 2.0, "pairs_r_dict": {}}

    optimize_params = {"cor_order": ("int", (2, 3)), "maxdeg": ("int", (4, 8))}
    basis_dependency_source_target = ("cor_order", "maxdeg")
    do_HAL_test(None, fixed_basis_info, optimize_params, basis_dependency_source_target, fit_data, monkeypatch, tmp_path, T_K=(100, 1000))

    # make sure cell changed at least a little
    traj = []
    for f in glob.glob("test_HAL.traj.it_*.extxyz"):
        ats = ase.io.read(f, ":")
        if len(ats) > len(traj):
            traj = ats
    assert len(traj) > 100
    assert np.mean([at.get_kinetic_energy() for at in traj[-25:]]) / np.mean([at.get_kinetic_energy() for at in traj[10:10+25]]) > 3.0


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

    new_fit_configs, basis_info, new_test_configs = HAL(
            fit_configs, fit_configs, basis_source, solver,
            fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights},
            n_iters=n_iters, traj_len=2000, tol=0.4, tol_eps=0.1, tau_rel=0.2, 
            ref_calc=EMT(), dt=1.0, T_K=T_K, T_tau=100, P_GPa=P_GPa,
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
        assert l > 1 and l <= 201
        print("do_HAL_test got len(traj)", l)
