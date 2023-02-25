import sys
import time

from pprint import pformat
from pathlib import Path

import numpy as np

import ase.io
from ase.md.langevin import Langevin

from ACEHAL.fit import fit
from ACEHAL.basis import define_basis
from ACEHAL.bias_calc import BiasCalculator, TauRelController
from ACEHAL.optimize_basis import optimize, estimate_dists_per_pair

from ACEHAL.dyn_utils import CellMC, HALMonitor, HALTolExceeded
from ACEHAL import viz

def HAL(fit_configs, traj_configs, basis_source, solver, fit_kwargs,
        n_iters, traj_len, tol, tol_eps, tau_rel, ref_calc, dt, T_K, T_tau,
        P_GPa=None, cell_step_interval=10, cell_step_mag=0.01,
        tau_hist=100, default_basis_info=None, basis_optim_kwargs=None, basis_estimate_dists="min", basis_optim_interval=None,
        file_root=None, traj_interval=10, test_fraction=0.0):
    """Iterate with hyperactive learning

    Parameters
    ----------
    fit_configs: list(Atoms)
        initial fitting configs
    traj_configs: list(Atoms)
        configs to start trajectories from
    basis_source: str 
        module with basis or text string with julia source to construct basis
    solver: sklearn-compatible LinearSolver
        solvers for model parameter design matrix linear problem
    fit_kwargs: dict
        User (not HAL provided) keyword args for `fit.fit()`, in particular "E0s", "data_keys", and "weights".
        HAL will provide "atoms_list", "solver", "B_len_norm", and "return_linear_problem".
        NOTE: this function explicitly uses "data_keys" and "E0s" fit_kwargs entries - should those be passed in explicitly? 
            If so, should HAL add them to fit_kwargs, or should the caller?
    n_iters: int
        number of HAL iterations
    traj_len: int
        max len of trajectories (unless tau_rel exceeded)
    tol: float
        tolerance for triggering HAL in fractional force error. If negative, save first config that 
        exceeds tol but continue trajectory to full traj_len
    tol_eps: float
        regularization epsilon to add to force denominator when computing relative force error for HAL tolerance
    tau_rel: float
        strength of bias forces relative to unbiased forces
    ref_calc: Calculator
        calculator for reference quantities, or None for dry run
    dt: float
        time step for dynamics
    T_K: float / tuple(float, float)
        temperature (in K) for Langevin dynamics, fixed or range for ramp
    T_tau: float
        time scale for Langevin dynamics friction
    P_GPa: float / tuple(float, float), default None
        pressure (in GPa) for dynamics, fixed or range for ramp, None for fixed cell
    cell_step_interval: int, default 25
        interval for attempts of Monte Carlo cell steps
    cell_step_mag: float, default 0.01
        magnitude of perturbations for cell MC steps
    tau_hist: int, default 100
        length of time over which to smooth force magnitudes for HAL criterion tau_rel
    default_basis_info: dict, default None
        default parameters for basis (ignored if optimizing basis)
    basis_optim_kwargs: dict, default None
        User provided keyword arguments used to optimize basis with for `optimize_basis.optimize()`, 
        or None for no optimization. Usually includes "n_trials", "fixed_basis_info", "optimize_params",
        "max_basis_len", and perhaps "score" or "seed".  Arguments that will be provided by HAL are "solver",
        "fitting_db", "basis_kwargs", and "fit_kwargs".
    basis_estimate_dists: "min" / "mean" / False
        If not False, estimate distances ("r_in_pairs", "r_in_elems", "r_in_global", and same for "r_0_*")
        from data, and string value indicates method for estimating "r_0_global"
    basis_optim_interval: int, default None
        interval (in HAL iterations, whether or not they generate configs added to fitting database)
        between re-optimizing basis, None to only optimize in initial step.
    file_root: str / Path, default None
        base part of path to all saved files (trajectory, potential, new config, plots)
    traj_interval: int, default 10
        interval between trajectory snapshots
    test_fraction: float, default 0.0
        fraction of configs to select (stochastic) for testing set

    Returns
    -------
    new_configs: list(Atoms) with configurations added by HAL
    basis_info: dict with info for constructing last basis used
    if test_fraction > 0.0:
        test_configs: list(Atoms) with test set configurations
    """

    assert basis_estimate_dists in ["min", "mean", False]

    # prepare file root of type Path
    if file_root is None:
        file_root = Path()
    file_root = Path(file_root)
    if file_root.is_dir():
        file_root = file_root / "HAL"

    def _HAL_label(it):
        n_dig = int(np.log10(n_iters)) + 1
        s = "it_{it:0" + str(n_dig) + "d}"
        return s.format(it=it)

    # initial basis
    if default_basis_info is not None:
        basis_info = default_basis_info.copy()
        B_len_norm = define_basis(default_basis_info, basis_source)
    elif basis_optim_kwargs is not None:
        t0 = time.time()
        basis_info = _optimize_basis(fit_configs, basis_source, solver, fit_kwargs, basis_optim_kwargs, basis_estimate_dists)
        B_len_norm = define_basis(basis_info, basis_source)
        print("TIMING initial_basis_optim", time.time() - t0)
    else:
        raise ValueError("One of default_basis_info and basis_optim_kwargs must be provided")

    print("HAL initial basis", basis_info)

    # initial fit
    t0 = time.time()
    committee_calc = _fit(fit_configs, solver, fit_kwargs, B_len_norm, file_root, _HAL_label(0))
    print("TIMING initial_fit", time.time() - t0)
    sys.stdout.flush()

    # prepare lists for new configs
    new_fit_configs = []
    if test_fraction > 0.0:
        new_test_configs = []

    for iter_HAL in range(n_iters):
        HAL_label = _HAL_label(iter_HAL)

        # pick config to start trajectory from
        traj_config = traj_configs[iter_HAL % len(traj_configs)].copy()
        # delete existing reference calculation fields from info and arrays
        for d in [traj_config.info, traj_config.arrays]:
            for k in list(d.keys()):
                if k in fit_kwargs['data_keys'].values():
                    del d[k]
        traj_config.calc = BiasCalculator(committee_calc, 0.0)
        tau_rel_control = TauRelController(tau_rel, tau_hist)

        # attachment to monitor HAL tolerance, save data and annotated trajectory
        if traj_interval > 0:
            traj_filename = file_root.parent / (file_root.name + f".traj.{HAL_label}.extxyz")
        else:
            traj_filename = None
        hal_monitor = HALMonitor(traj_config, tol, tol_eps, tau_rel_control, traj_file=traj_filename, traj_interval=traj_interval)

        # set up T and P ramps
        if isinstance(T_K, (tuple, list)) or isinstance(P_GPa, (tuple, list)):
            n_stages = 20
        else:
            n_stages = 1
        if isinstance(T_K, (tuple, list)):
            assert len(T_K) == 2
            ramp_Ts = np.linspace(T_K[0], T_K[1], n_stages)
        else:
            ramp_Ts = [T_K] * n_stages
        if isinstance(P_GPa, (tuple, list)):
            assert len(P_GPa) == 2
            ramp_Ps = np.linspace(P_GPa[0], P_GPa[1], n_stages)
        else:
            ramp_Ps = [P_GPa] * n_stages

        t0 = time.time()
        # run dynamics for entire T ramp
        try:
            for T_K_cur, P_GPa_cur in zip(ramp_Ts, ramp_Ps):
                if P_GPa_cur is not None:
                    # attachment to do cell steps (inside ramp loop since it needs T_K_cur for MC acceptance criterion)
                    # E = N B strain^2
                    # strain = sqrt(E / (B N))
                    # aim for cell_step_mag for 500 K and 40 atoms
                    cell_step_mag_use = cell_step_mag * np.sqrt((T_K_cur / 500.0) / (len(traj_config) / 40))
                    cell_mc = CellMC(traj_config, T_K_cur, P_GPa_cur, cell_step_mag_use)
                else:
                    cell_mc = None

                # set up dynamics for this section of ramp
                dyn = Langevin(traj_config, dt, temperature_K=T_K_cur, friction=1.0 / T_tau)

                # attach monitor and cell steps
                dyn.attach(hal_monitor)
                if cell_mc is not None:
                    dyn.attach(cell_mc, interval=cell_step_interval)

                # run trajectory for this section of T ramp
                dyn.run(traj_len // len(ramp_Ts))
                sys.stdout.flush()

                # mark restart so next call from next dyn.run skips first config
                hal_monitor.mark_restart()

        except HALTolExceeded:
            pass
        print("TIMING trajectory", time.time() - t0)

        # save HAL-selected config
        if hal_monitor.HAL_trigger_config is not None:
            new_config = hal_monitor.HAL_trigger_config
        else:
            # no trigger, just grab most recent config
            new_config = traj_config

        # If trajectory was triggered by HAL tolerance, config was not already written
        # and will be written here.  If trajectory completed normally and last config was
        # already set to be written according to traj_interval, it was already written and
        # this call should know and not actually write it again. 
        hal_monitor.write_final_config(new_config)

        plot_traj_file = file_root.parent / (file_root.name + f".run_data.{HAL_label}.pdf")
        trigger_data = {"criterion": (hal_monitor.HAL_trigger_step, np.abs(tol))}
        viz.plot_HAL_traj_data(hal_monitor.run_data, trigger_data, plot_traj_file)

        print(f"HAL iter {iter_HAL} got config with criterion {new_config.info['HAL_criterion']} at time step {hal_monitor.HAL_trigger_step} / {traj_len}")
        sys.stdout.flush()

        if ref_calc is not None:
            t0 = time.time()
            # do reference calculation
            new_config.calc = ref_calc

            data_keys = fit_kwargs['data_keys']
            if 'E' in data_keys:
                E = new_config.get_potential_energy(force_consistent=True)
                new_config.info[data_keys['E']] = E
            if 'F' in data_keys:
                F = new_config.get_forces()
                new_config.new_array(data_keys['F'], F)
            if 'V' in data_keys:
                V = - new_config.get_volume() * new_config.get_stress(voigt=False)
                new_config.info[data_keys['V']] = V
            print("TIMING reference_calc", time.time() - t0)

        # save new config to fit or test set
        rv = np.random.uniform() 
        if rv > test_fraction:
            # fit config chosen
            new_fit_configs.append(new_config)
            new_config_file = file_root.parent / (file_root.name + f".config_fit.{HAL_label}.extxyz")

            if ref_calc is not None:
                # cause a refit below, whether or not basis is re-optimized
                committee_calc = None
        else:
            # test config chosen, not need for refit
            new_test_configs.append(new_config)
            new_config_file = file_root.parent / (file_root.name + f".config_test.{HAL_label}.extxyz")

        ase.io.write(new_config_file, new_config)

        if (basis_optim_kwargs is not None and basis_optim_interval is not None and
            iter_HAL % basis_optim_interval == basis_optim_interval - 1):
            t0 = time.time()
            # optimize basis
            basis_info = _optimize_basis(fit_configs + new_fit_configs, basis_source, solver, fit_kwargs,
                                         basis_optim_kwargs, basis_estimate_dists)
            print("HAL got optimized basis", basis_info)
            B_len_norm = define_basis(basis_info, basis_source)
            # reset calculator to trigger a it with the new basis based on the optimized basis_info
            committee_calc = None
            print("TIMING basis_optim", time.time() - t0)

        if committee_calc is None:
            t0 = time.time()
            # re-fit (whether because of new config or new basis or both)
            # label potential with next iteration, since that's when it will be used
            committee_calc = _fit(fit_configs + new_fit_configs, solver, fit_kwargs, B_len_norm, file_root, _HAL_label(iter_HAL + 1))
            print("TIMING fit", time.time() - t0)

    # return fit configs, final basis_info, and optionally test configs
    if test_fraction > 0.0:
        return new_fit_configs, basis_info, new_test_configs
    else:
        return new_fit_configs, basis_info


def _estimate_dists(configs, basis_optim_kwargs, basis_estimate_dists):
    """estimate r_0 and r_in, and store per-pair, per-element, and
    global values in existing or new basis_optim_kwargs["fixed_basis_info"]

    Modifies basis_optim_kwargs["fixed_basis_info"]

    Parameters
    ----------
    configs: list(Atoms)
        atomic configs to get dists from
    basis_optim_kwargs: dict
        keyword args for optimize_basis, in particular "fixed_basis_info"
    basis_estimate_dists: "min" / "mean"
        method to compute per-element and global r_0 from per pair value (r_in is always min)
    """
    assert basis_estimate_dists in ["min", "mean"]

    # raw per-pair distances
    r_in_pairs, r_0_pairs = estimate_dists_per_pair(configs)

    # NOTE: not obvious if reduction to per elemenet and global values should use
    # each species pair once (as it is now), or once as (s0, s1) and once as (s1, s0). 
    # Possibly they should actually be derived from separately computed RDFs with
    # appropriate filtering.

    if basis_estimate_dists == "min":
        r_0_global_func = np.min
    else:
        r_0_global_func = np.mean

    # per-element values from min (r_in) or mean (r_0) over pairs that contain each element
    syms = set([sym for sym, _ in r_in_pairs] + [sym for _, sym in r_in_pairs])
    r_in_elems = {sym: np.min([v for k, v in r_in_pairs.items() if sym in k]) for sym in syms}
    r_0_elems = {sym: r_0_global_func([v for k, v in r_0_pairs.items() if sym in k]) for sym in syms}

    # global values from min (r_in) and selected method (r_0) over all pairs
    r_in_global = np.min(list(r_in_pairs.values()))
    r_0_global = r_0_global_func(list(r_0_pairs.values()))

    fixed_basis_info = basis_optim_kwargs.get("fixed_basis_info", {})

    fixed_basis_info["pairs_r_dict"] = {sym_pair: {"r_in": r_in_pairs[sym_pair], "r_0": r_0_pairs[sym_pair]} for sym_pair in r_in_pairs}
    fixed_basis_info["elems_r_dict"] = {sym: {"r_in": r_in_elems[sym], "r_0": r_0_elems[sym]} for sym in r_in_elems}
    fixed_basis_info["r_0"] = r_0_global
    fixed_basis_info["r_in"] = r_in_global

    print("HAL estimated dists r_pairs", pformat(fixed_basis_info["pairs_r_dict"]))
    print("HAL estimated dists r_elems", pformat(fixed_basis_info["elems_r_dict"]))
    print("HAL estimate dists global r_in", r_in_global, "r_0", r_0_global)

    if "fixed_basis_info" not in basis_optim_kwargs:
        # if it's a new field, add it
        basis_optim_kwargs["fixed_basis_info"] = fixed_basis_info


def _optimize_basis(fit_configs, basis_source, solver, fit_kwargs, basis_optim_kwargs, basis_estimate_dists):
    """Optimize a basis

    Parameters
    ----------
    fit_configs: list(Atoms)
        atomic configurations to fit
    basis_source: str
        source for basis (module or julia source code)
    solver: LinearSolver
        solver for linear problem
    fit_kwargs: dict
        keyword args for fit()
    basis_optim_kwargs: dict
        keyword args for optimize_basis()
    basis_estimate_dists: "min" / "mean" / False, default "min"
        If not False, estimate distances ("r_in_pairs", "r_in_elems", "r_in_global", and same for "r_0_*")
        from data, and string value indicates method for estimating "r_0_global"

    Returns
    -------
    basis_info dict with parameters that can be passed to define_basis()
    """
    if basis_estimate_dists:
        _estimate_dists(fit_configs, basis_optim_kwargs, basis_estimate_dists)

    # do the optimization
    basis_info = optimize(solver=solver, fitting_db=fit_configs, 
            basis_kwargs={"julia_source": basis_source},
            fit_kwargs=fit_kwargs, **basis_optim_kwargs)

    return basis_info


def _fit(fit_configs, solver, fit_kwargs, B_len_norm, file_root, HAL_label):
    """Do a fit

    Parameters
    ----------
    fit_configs: list(Atoms)
        fitting config
    solver: sklearn LinearSolver
        solver for linear problem
    fit_kwargs: dict
        other arguments for fit.fit()
    B_len_norm: tuple(basis, int, vector / None)
        basis, its length, and optional normalization vector returned by define_basis
    file_root: str / Path
        root part of path used to save potential
    HAL_label: str
        label added to file_root for filenames

    Returns
    -------
    committee_calc ACECommitteeCalc
    """
    # no calculator defined, fit one with the current fitting configs and basis
    pot_filename = str(file_root.parent / (file_root.name + f".pot.{HAL_label}.json"))

    committee_calc = fit(atoms_list=fit_configs, solver=solver, B_len_norm=B_len_norm,
                         return_linear_problem=False, pot_file=pot_filename, **fit_kwargs)

    plot_dimers_file = file_root.parent / (file_root.name + f".dimers.{HAL_label}.pdf")
    viz.plot_dimers(committee_calc, list(fit_kwargs["E0s"]), plot_dimers_file)

    return committee_calc
