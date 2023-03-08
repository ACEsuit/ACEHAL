import numpy as np

import pandas as pd

from ase.constraints import full_3x3_to_voigt_6_stress

#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

from julia.JuLIP import energy, forces, virial
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")
ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")

from .ace_committee_calc import ACECommittee


def fit(atoms_list, solver, B_len_norm, E0s, data_keys, weights, Fmax=None, n_committee=8,
        rng=None, pot_file=None, data_save_label=None, return_linear_problem=False, report_errors=True,
        verbose=False):
    """Fit an ACE model with a committee from a list of Atoms

    Parameters
    ----------
    atoms_list: list(Atoms)
        atomic configurations to fit
    solver: sklearn-compatible LinearSolver
        solvers for the linear problem
    B_len_norm: (julia basis object, int, array(float) / None)
        3-tuple representing basis to use, as returned by ACEHAL.basis.define_basis,
        consisting of basis object, integer length (ignored), and optional array with 
        normalization factors.
    E0s: dict{str: float}
        dict of atomic energies for each species
    data_keys: dict{'E' / 'F' / 'V': str}
        dict with Atoms.info (energy, virial) and Atoms.arrays (forces) keys
    weights: dict{'E' / 'F' / 'V' / 'E_per_atom' / 'V_per_atom': float}
        default weights for each property in the fitting
    Fmax: float, default None
        max force magnitude above which to drop force from fitting problem
    n_committee: int, default 8
        number of members in committee
    rng: numpy Generator, default None
        random number generator to use, or np.random if None
    pot_file: str / Path, default None
        optional file to save potential to
    data_save_label: str, default None
        optional label for files to save design matrix, RHS, coefficients, etc to
    return_linear_problems: bool, default False
        optionally return matrices for linear problem
    report_errors: bool / list((str, list(Atoms)), ...), default True
        If False, report no errors.  If True, report errors on fitting db,
        otherwise list of tuples with labels and sets of configs to report errors on.
    verbose: bool, default False
        verbose output

    Returns
    -------
    ACECommittee fit calculator with optional committee
    iff return_linear_problem is True:
        Psi, Y, coef: numpy array(float) design matrix, RHS and coefficients 
        prop_row_inds: dict('E' / 'F' / 'V': list(int)) with indices of Psi and Y rows corresponding to each type of property
    """
    Psi, Y, prop_row_inds = assemble_Psi_Y(atoms_list, B_len_norm[0], E0s, data_keys, weights, Fmax=Fmax)

    calc, coef = do_fit(Psi, Y, B_len_norm[0], E0s, solver, n_committee=n_committee, basis_normalization=B_len_norm[2],
                        pot_file=pot_file, rng=rng, verbose=verbose)

    if report_errors:
        try:
            for label, configs in report_errors:
                raise NotImplementedError("report_errors for arbitrary config sets not implemented yet")
        except TypeError:
            resid_sq = (Psi @ coef - Y) ** 2
            err_data = {"label": "fit"}
            err_data.update({p: [np.sqrt(np.mean(resid_sq[prop_row_inds[p]]))] for p in ['E', 'F', 'V']})
        df = pd.DataFrame(err_data)
        print("fitting residuals, Psi.shape", Psi.shape)
        print(df.to_string())

    if data_save_label is not None:
        args = {'file': data_save_label + ".Psi.npz",
                'Psi': Psi, 
                'Y': Y, 
                'c': coef}
        try:
            args['sigma'] = solver.sigma_
        except AttributeError:
            pass

        np.savez_compressed(**args)

    if return_linear_problem:
        results = (calc, Psi, Y, coef, prop_row_inds)
    else:
        results = calc
    return results


def _Psi_Y_section(at, B, E0s, data_keys, weights, Fmax=None):
    """Compute a section of the design matrix and RHS for a single atoms object

    Parameters
    ----------
    at: ASE Atoms
        atomic configuration. Fit quantities in Atoms.info/Atoms.arrays
        with keys specified by data_keys
    B: basis
        basis object returned from julia.Main
    E0s: dict{str: float}
        dict with atomic energy offsets for each species
    data_keys: dict{str: str}
        dict with Atoms.info/Atoms.arrays key for 'E', 'F', 'V'
    weights: dict('E' / 'F' / 'V' / 'E_per_atom' / 'V_per_atom': float)
        weights of each quantity type. Multiplied by per-config
        and (for forces) per-atom weights in Atoms.info or Atoms.arrays,
        respective, fields named data_keys[prop] + "_weight"
    Fmax: float, default None
        max force magnitude above which to ignore
        NOTE: should this be a more general mask?

    Returns
    -------
    Psi: numpy array (N_data, N_basis) design matrix
    Y: numpy array (N_data) right hand side
    prop_row_inds: dict('E' / 'F' / 'V' : list(int)) indices of rows corresponding to E, F, and V quantities
    """
    Psi = []
    Y = []
    prop_row_inds = {'E': [], 'F': [], 'V': []}

    if data_keys.get("E") in at.info:
        # N_B
        E_B = np.array(energy(B, convert(ASEAtoms(at))))
        if np.any(np.isnan(E_B)):
            import sys, ase.io
            ase.io.write(sys.stderr, at, format="extxyz")
            raise ValueError("NaN constructing design matrix for energy " + str(np.isnan(E_B)))
        if "E" in weights:
            weight_E = weights["E"]
        elif "E_per_atom" in weights:
            weight_E = weights["E_per_atom"] / len(at)
        else:
            raise ValueError("Need E or E_per_atom in weights")
        weight_E *= at.info.get(data_keys["E"] + "_weight", 1.0)
        Psi.append(weight_E * E_B)
        Y.append(weight_E * (at.info[data_keys["E"]] - np.sum([at.symbols.count(sym) * E0 for sym, E0 in E0s.items()])))

        prop_row_inds['E'].append(0)

    if data_keys.get("F") in at.arrays:
        # N_B x N_atoms x 3
        F_B = np.array(forces(B, convert(ASEAtoms(at))))
        if np.any(np.isnan(F_B)):
            import sys, ase.io
            ase.io.write(sys.stderr, at, format="extxyz")
            raise ValueError("NaN constructing design matrix for forces " + str(np.isnan(F_B)))

        # filter F <= Fmax
        F = at.arrays[data_keys["F"]]
        if Fmax is not None:
            F_filter = np.linalg.norm(F, axis=1) <= Fmax
        else:
            F_filter = [True] * len(F)
        F = F[F_filter, :]

        # N_B x (N_atoms with not too large F) x 3
        F_B = F_B[:, F_filter, :]

        # N_B x (N_atoms with not too large F) * 3
        F_B = F_B.reshape((F_B.shape[0], -1))

        per_config_weight = at.info.get(data_keys["F"] + "_weight", at.info.get(data_keys["F"] + "_weight", 1.0))
        per_atom_weight = at.arrays.get(data_keys["F"] + "_weight", at.arrays.get(data_keys["F"] + "_weight", np.ones(len(at))))
        per_atom_weight = per_atom_weight[F_filter]
        if len(per_atom_weight.shape) == 1 or per_atom_weight.shape[1] == 1:
            per_atom_weight = np.repeat(per_atom_weight, 3)
        weight_F = (weights["F"] * per_config_weight * per_atom_weight)
        Psi.extend(weight_F[:, np.newaxis] * F_B.T)
        Y.extend(weight_F * F.reshape((-1)))

        prop_row_inds['F'].extend(np.arange(len(Y) - F.size, len(Y)))

    if data_keys.get("V") in at.info:
        # N_B x 3 x 3
        V_B = np.array(virial(B, convert(ASEAtoms(at))))
        if np.any(np.isnan(V_B)):
            import sys, ase.io
            ase.io.write(sys.stderr, at, format="extxyz")
            raise ValueError("NaN constructing design matrix for virial " + str(np.isnan(V_B)))

        # select 6 independent Voigt elements of V
        # note that V might come in as (9,) or (3,3), so reshape first
        V = full_3x3_to_voigt_6_stress(at.info[data_keys["V"]].reshape((3, 3)))

        # N_B x 6
        V_B = np.asarray([full_3x3_to_voigt_6_stress(V_B[basis_i, :, :]) for basis_i in range(V_B.shape[0])])

        if "V" in weights:
            weight_V = weights["V"]
        elif "V_per_atom" in weights:
            weight_V = weights["V_per_atom"] / len(at)
        else:
            raise ValueError("Need V or V_per_atom in weights")
        weight_V *= at.info.get(data_keys["V"] + "_weight", 1.0)
        Psi.extend(weight_V * V_B.T)
        Y.extend(weight_V * V)

        prop_row_inds['V'].extend(np.arange(len(Y) - V.size, len(Y)))

    return Psi, Y, prop_row_inds


def assemble_Psi_Y(ats, B, E0s, data_keys, weights, Fmax=None):
    """Assemble the entire design matrix, right hand side, and indices of E, F, V related rows

    Parameters
    ----------
    ats: list(ASE Atoms)
        atomic configurations. Fit quantities in Atoms.info/Atoms.arrays
        with keys specified by data_keys
    B: basis
        basis object returned from julia.Main
    E0s: dict{str: float}
        dict with atomic energy offsets for each species
    data_keys: dict{str: str}
        dict with Atoms.info/Atoms.arrays key for 'E', 'F', 'V'
    weights: dict('E' / 'F' / 'V' / 'E_per_atom' / 'V_per_atom': float)
        weights of each quantity type. Multiplied by per-config
        and (for forces) per-atom weights in Atoms.info or Atoms.arrays,
        respective, fields named data_keys[prop] + "_weight"
    Fmax: float, default None
        max force magnitude above which to ignore
        NOTE: should this be a more general mask?

    Returns
    -------
    Psi: numpy array (N_data, N_basis) design matrix
    Y: numpy array (N_data) right hand side
    prop_row_inds: dict('E' / 'F' / 'V' : list(int)) indices of rows corresponding to E, F, and V quantities
    """
    Psi = []
    Y = []
    prop_row_inds = {'E': [], 'F': [], 'V': []}
    last_Y_len = 0
    for at in ats:
        Psi_sec, Y_sec, prop_row_inds_sec = _Psi_Y_section(at, B, E0s, data_keys, weights, Fmax=Fmax)
        Psi.extend(Psi_sec)
        Y.extend(Y_sec)
        for p in prop_row_inds:
            prop_row_inds[p].extend([ind + last_Y_len for ind in prop_row_inds_sec[p]])
        last_Y_len = len(Y)

    return np.asarray(Psi), np.asarray(Y), prop_row_inds


def do_fit(Psi, Y, B, E0s, solver, n_committee=32, basis_normalization=None, pot_file=None, rng=None, verbose=False):
    """fit an ACE committee model to a design matrix and RHS

    Parameters
    ----------
    Psi: numpy array (n_data, n_basis) float
        design matrix
    Y: numpy arrays (n_data) float
        right hand side
    solver: numpy LinearModel
        solver to use, typically BayesianRidge, ARDRegression, or BayesRegressionMax
    n_committee: int, default 8
        number of members in committee
    basis_normalization: numpy array (n_basis) float, default None
        normalization of design matrix columns, e.g. for smoothness prior
    pot_file: str / Path, default None
        optional file to save potential to
    rng: numpy Generator, default None
        random number generator to use, or np.random if None
    verbose: bool, default False
        verbose output

    Returns
    -------
    model ACECommittee 
    c coefficients vector
    """
    if verbose:
        print("fitting with design matrix shape", Psi.shape)

    # normalize basis
    if basis_normalization is not None:
        assert basis_normalization.shape == (Psi.shape[1],)
        Psi_norm = Psi / basis_normalization
    else:
        Psi_norm = Psi

    solver.fit(Psi_norm, Y)

    c_norm = solver.coef_

    # undo normalization in coefficients, so users of solver outside this function will
    # get consistent ones
    if basis_normalization is not None:
        c = solver.coef_ / basis_normalization
    else:
        c = solver.coef_

    if verbose:
        print("fitting got nonzero coeffs", len(np.nonzero(c)[0]))
        try:
            print("fitting got scores", solver.scores_)
        except:
            print("fitting got scores", None)
        # print("fitting got |Psi @ coeff - y|", np.linalg.norm(Psi @ c - Y))
        print("fitting got RMS Psi @ coeff - y", np.sqrt(np.mean((Psi @ c - Y)**2)))

    if n_committee > 0:
        sigma = solver.sigma_

        # sklearn ARDRegression solver returns sigma only for selected features, but does not explicitly
        # indicate which ones those are.
        if sigma.shape[0] != len(c_norm):
             # only valid for sklearn ARDRegression
            included_c = solver.lambda_ < solver.threshold_lambda
            assert sigma.shape[0] == sum(included_c)

            sigma_full = np.zeros((len(c_norm), len(c_norm)), dtype=sigma.dtype)
            inds = np.where(included_c)[0]
            sigma_full[inds[:, None], inds] += sigma

            sigma = sigma_full
    else:
        sigma = None

     # create committee coefficients
    if n_committee > 0:
        # make sure that raw coefficients (c_norm) are used, since sigma is still scaled to be
        # consistent with those, and unscale resulting committee coefficients below
        if rng is None:
            comms = np.random.multivariate_normal(c_norm, sigma, size=n_committee)
        else:
            comms = rng.multivariate_normal(c_norm, sigma, size=n_committee)

        if basis_normalization is not None:
            # also undo normalization on committee coefficients
            comms /= basis_normalization
    else:
        comms = None

    Main.E0s = E0s
    Main.ref_pot = Main.eval("refpot = OneBody(" + "".join([" :{} => {}, ".format(key, value) for key, value in E0s.items()]) + ")")
    Main.B = B
    Main.c = c
    if comms is not None:
        Main.comms = comms

    IP_mean = Main.eval("ACE_IP = JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, c))")
    if comms is not None:
        IP_committee = Main.eval("COMMITTEE_IP = JuLIP.MLIPs.SumIP(ref_pot, ACE1.committee_potential(B, c, transpose(comms)))")
    else:
        IP_committee = None

    if pot_file is not None:
        Main.eval(f'save_dict("{pot_file}", Dict("IP" => write_dict(ACE_IP)))')

    committee_calc = ACECommittee("ACE_IP", "COMMITTEE_IP" if comms is not None else None)
    return committee_calc, c
