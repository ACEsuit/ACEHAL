import numpy as np

from scipy.signal import argrelextrema

import ase.data
from matscipy.neighbours import neighbour_list as neighbor_list

import optuna 
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError

from .basis import define_basis
from .fit import fit

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


def basis_dependency_range_max(basis_kwargs, fixed_basis_info, optimize_params, max_basis_len, dependency_source, dependency_target):
    """Make the max of range of optimization values of the dependency_target be dependent on the 
    value of the dependency_source so that the max basis length is not exceeded

    Modifies optimize_params (keeping all other parameters at their minimum values) so that range 
    maximum of dependency_target ensures that max basis length is not exceeded.

    Parameters
    ----------
    basis_kwargs: dict
        parameters to `basis.define_basis()` other than basis_info
    fixed_basis_info: dict
        parameters for basis_info that are not optimized
    optimize_params: dict
        parameters that 
    max_basis_len: int
        max basis length to allow
    dependency_source: str
        key for params that controls dependency
    dependency_target: str
        key for param whose range max is set by dependency
    """
    assert optimize_params[dependency_source][0] == "int"
    assert optimize_params[dependency_target][0] == "int"

    source_range = optimize_params[dependency_source][1]
    optimize_params_target = {}
    for source_val in range(source_range[0], source_range[1] + 1):
        min_target_val = optimize_params[dependency_target][1][0]

        target_val = min_target_val
        basis_len = 0
        while basis_len <= max_basis_len:
            basis_info = fixed_basis_info.copy()
            for param in optimize_params:
                basis_info[param] = optimize_params[param][1][0]
            basis_info[dependency_source] = source_val
            basis_info[dependency_target] = target_val
            _, basis_len, _ = define_basis(basis_info=basis_info, **basis_kwargs)

            if basis_len <= max_basis_len:
                target_val += 1

        optimize_params_target[source_val] = (min_target_val, target_val - 1)

    optimize_params[dependency_target] = ("int", (dependency_source, optimize_params_target))

    source_range = optimize_params[dependency_source][1]
    min_source_val = source_range[1] + 1
    max_source_val = source_range[0] - 1
    for source_val in range(source_range[0], source_range[1] + 1):
        target_range = optimize_params[dependency_target][1][1][source_val]
        if target_range[1] >= target_range[0]:
            # range is not empty
            min_source_val = min(min_source_val, source_val)
            max_source_val = max(max_source_val, source_val)
        else:
            del optimize_params[dependency_target][1][1][source_val]
    optimize_params[dependency_source] = (optimize_params[dependency_source][0], (min_source_val, max_source_val))


class BasisTooLarge(Exception):
    pass

class StopWhenTrialKeepFailingCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive_failed_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.FAIL:
            self._consecutive_failed_count += 1
        else:
            self._consecutive_failed_count = 0

        if self._consecutive_failed_count >= self.threshold:
            study.stop()

def optimize(solver, fitting_db, n_trials, optimize_params, basis_kwargs, fit_kwargs, fixed_basis_info=None, max_basis_len=None,
             score="BIC", timeout=600, addl_guesses=[], seed=None):
    """optimize the basis by maximizing a score over a number of optuna trials

    Parameters
    ----------
    solver: sklearn LinearSolver compatible object
        solver to pass into fit.fit()
    fitting_db: list(Atoms)
        fitting database to pass to fit.fit()
    n_trials: int
        number of trials in study
    optimize_params: dict(str: ("int" / "float", (num, num) / (str, dict(int: (num, num))) ))
        dict with information on params to optimize (keys) and for each a string
        indicating type and either a 2-tuple with the range or a 2-tuple with the
        name of the param the range depends on, and a dict with keys indicating
        the dependency param value and values with the range for that value.
        Example that sets a max ACE polynomial degree integer guess range of 2-10 for
        correlation order 2 and 2-6 for correlation order 3:
        { "maxdeg_ACE": ("int", ("cor_order": {2: (2, 10), 3: (2, 6)})) }
    basis_kwargs: dict
        keywords arguments to pass to basis.define_basis() in addition to basis_info
    fit_kwargs: dict
        keyword args for fit.fit() in addition to list of atoms, solver, basis returned
        from basis.define_basis, and return_linear_problem
    fixed_basis_info: dict, default {}
        dict with constant values for basis_info
    max_basis_len: int, default None
        max basis length
    score: str, default "BIC"
        method for calculating score: "BIC", "AIC", "AICc", and "solver_internal" (for
        solver.score_[-1])
    timeout: int, default 600
        max time allowed for each trial
    addl_guesses: list(dict), default []
        list of dicts containing parameters sets for additional guesses

    Returns
    -------
    basis_info dict that can be passed into basis.define_basis()
    """

    if fixed_basis_info is None:
        basis_info = {}
    else:
        basis_info = fixed_basis_info.copy()

    @timeout_decorator.timeout(timeout, use_signals=True)
    def objective(trial):
        for param, (param_type, param_range) in optimize_params.items():
            if isinstance(param_range[0], str):
                # get range from dependency
                param_range_dep_source = param_range[0]
                if param_range_dep_source not in basis_info:
                    raise ValueError(f"Got dependency of param {param} on {param_range_dep_source} but it is not yet defined in basis_info {basis_info.keys()}")

                range_dep_values = param_range[1]
                dep_source_value = basis_info[param_range_dep_source]
                param_range = range_dep_values[dep_source_value]

            if param_type == "int":
                suggest_func = trial.suggest_int
            elif param_type == "float":
                suggest_func = trial.suggest_float
            else:
                raise NotImplementedError(f"Unknown type {param_type} for param {param}")

            basis_info[param] = suggest_func(param, low=param_range[0], high=param_range[1])

        B_len_norm = define_basis(basis_info=basis_info, **basis_kwargs)
        trial.set_user_attr("B_len", B_len_norm[1])

        if max_basis_len is not None and B_len_norm[1] > max_basis_len:
            raise BasisTooLarge(f"basis {basis_info} len {B_len_norm[1]} > {max_basis_len}")

        if "report_errors" not in fit_kwargs:
            # don't report errors within basis optimization by default
            fit_kwargs_use = fit_kwargs.copy()
            fit_kwargs_use["report_errors"] = False
        else:
            fit_kwargs_use = fit_kwargs
        calc, Psi, Y, coef, _ = fit(fitting_db, solver, B_len_norm, return_linear_problem=True, **fit_kwargs_use)

        n = Psi.shape[0]

        if hasattr(solver, 'threshold_lambda'):
            included_c = solver.lambda_ < solver.threshold_lambda
            k = sum(included_c)
        else:
            k = Psi.shape[1]    

        print("k: ", k)
        print("Psi.shape[1]: ", Psi.shape[1])

        if score == "BIC":
            residuals = Psi @ coef - Y
            trial_score = n * np.log(np.mean(residuals ** 2)) + k * np.log(n)
        elif score == "AIC":
            residuals = Psi @ coef - Y
            trial_score = n * np.log(np.mean(residuals ** 2)) + 2 * k 
        elif score == "AICc":
            residuals = Psi @ coef - Y
            if n - k - 1 <= 0:
                raise ValueError(f"Ill-defined AICc when n - k - 1 = {n} - {k} - 1 = {n - k - 1} <= 0")
            trial_score = n * np.log(np.mean(residuals ** 2)) + 2 * k + (2 * k ** 2 + 2 * k) / (n - k - 1)
        elif score == "solver_internal":
            trial_score = - solver.scores_[-1]
        else:
            raise ValueError(f"Unknown score method {score}")

        return trial_score

    study = optuna.create_study(sampler=TPESampler(seed=seed), direction='minimize')

    for guess in addl_guesses:
        guess_dict = guess.copy()
        for k in fixed_basis_info.keys():
            del guess_dict[k]
        study.enqueue_trial(guess_dict)

    study.optimize(objective,
                   callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)),
                              StopWhenTrialKeepFailingCallback(n_trials * 2)],
                   catch=(TimeoutError, BasisTooLarge))

    basis_info = fixed_basis_info.copy()
    basis_info.update(study.best_params)

    print(f"BEST BASIS params {study.best_params} attrib {study.best_trial.user_attrs} score {study.best_value}")

    return basis_info
