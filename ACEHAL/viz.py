from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

import pandas as pd

from ase.atoms import Atoms
import ase.data

from ACEHAL.bias_calc import BiasCalculator

from ase.constraints import full_3x3_to_voigt_6_stress


def error_table(config_sets, calc, data_keys, Fmax=10.0):
    """Create a pandas DataFrame error table

    Parameters
    ----------
    config_sets: list((str, list(Atoms)))
        list of labels 
    calc: calculator
        calculator to use
    data_keys: dict(str: str)
        dict of keys (values) to use for reference energy ("E"), force ("F"), and virial ("V")
    Fmax: float, default 10.0
        max force - if exceeded, exclude that atom's force as well as entire configurations energy and virial
        (like fit)

    Returns
    -------
    df: pandas.DataFrame with report
    """
    err_data = {"E/at": [], "F": [], "V/at": []}
    index = []

    if isinstance(config_sets[0], Atoms):
        config_sets = [config_sets]

    for label_atoms in config_sets:
        if isinstance(label_atoms[0], Atoms):
            # bare Atoms
            label = None
            atoms_list = label_atoms
        elif len(label_atoms[1]) == 0 or isinstance(label_atoms[1][0], Atoms):
            # list of Atoms
            label, atoms_list = label_atoms
        else:
            raise ValueError("Got config_set containing something other than list(Atoms) or (str, list(Atoms))")

        if len(atoms_list) == 0:
            index.append(f"{label}")
            err_data["E/at"].append(np.nan)
            err_data["F"].append(np.nan)
            err_data["V/at"].append(np.nan)

            continue

        E_err = {"ALL": []}
        F_err = {"ALL": []}
        V_err = {"ALL": []}
        for at in atoms_list:
            error_group = at.info.get("error_group")
            at.calc = calc

            F = at.get_forces()
            if Fmax is not None:
                F_exceeded = np.linalg.norm(F, axis=1) > Fmax
            else:
                F_exceeded = [False] * len(at)

            if data_keys["E"] in at.info and not np.any(F_exceeded):
                E_err_item = (at.get_potential_energy() - at.info[data_keys["E"]]) / len(at)
                E_err["ALL"].append(E_err_item)
                if error_group is not None:
                    if error_group not in E_err:
                        E_err[error_group] = []
                    E_err[error_group].append(E_err_item)
            if data_keys["F"] in at.arrays:
                F_err_item = (F - at.arrays[data_keys["F"]])[~F_exceeded].reshape((-1))
                F_err["ALL"].extend(F_err_item)
                if error_group is not None:
                    if error_group not in F_err:
                        F_err[error_group] = []
                    F_err[error_group].extend(F_err_item)
            if data_keys["V"] in at.info and not np.any(F_exceeded):
                V_err_item = full_3x3_to_voigt_6_stress(- at.get_volume() * at.get_stress(voigt=False) -
                                                        at.info[data_keys["V"]]) / len(at)
                V_err["ALL"].extend(V_err_item)
                if error_group is not None:
                    if error_group not in V_err:
                        V_err[error_group] = []
                    V_err[error_group].extend(V_err_item)

        for error_group in E_err:
            if label is None:
                index.append(error_group)
            else:
                index.append(label + " " + error_group)
            E_err_group = np.asarray(E_err[error_group])
            F_err_group = np.asarray(F_err[error_group])
            V_err_group = np.asarray(V_err[error_group])
            err_data["E/at"].append(np.sqrt(np.mean(E_err_group ** 2)))
            err_data["F"].append(np.sqrt(np.mean(F_err_group ** 2)))
            err_data["V/at"].append(np.sqrt(np.mean(V_err_group ** 2)))

    return pd.DataFrame(err_data, index=index)


def plot_HAL_traj_data(run_data, trigger_data, plot_file, log_y=["criterion"]):
    """Plot the data gathered over the HAL trajectory

    Parameters
    ----------
    run_data: dict(str: list(float))
        list of quantities to plot, e.g. PE, KE, criterion, etc
    trigger_data, dict(str, (float, float))
        dict of x, y pairs to use for horizontal and vertical line 
        marking values of quantities at trigger
    plot_file: str / Path
        file to save plot to
    """
    fig = Figure(figsize=(6.0, 2.5 * len(run_data)))
    for panel_i, (key, data) in enumerate(run_data.items()):
        ax = fig.add_subplot(len(run_data) + 1, 1, panel_i + 1)
        if key in log_y:
            ax.set_yscale("log")
        ax.set_ylabel(key)
        ax.plot(range(len(data)), data, color='black')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if key in trigger_data:
            extra_x = trigger_data[key][0]
            extra_y = trigger_data[key][1]
            if extra_x is not None:
                # plot time of trigger if it was provided
                ax.plot([extra_x] * 2, ylim, '-', color='red')
            if (extra_y is not None and extra_y < max(data) + (max(data) - min(data)) and
                                        extra_y > min(data) - (max(data) - min(data))):
                # plot value of trigger if it was provided and within
                # a reasonable distance of range
                ax.plot(xlim, [extra_y] * 2, '-', color='red')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # xlabel only on bottom panel
    ax.set_xlabel("time step")
    fig.savefig(plot_file, bbox_inches='tight')

    plot_file = Path(plot_file)
    np.savez_compressed(plot_file.parent / (plot_file.name + ".npz"), **run_data)


def plot_dimers(calc, elements, plot_file, max_E_range=(-5.0, 5.0), r_range=(0.5, 10)):
    """Plot E(r) for every possible dimer
    Parameters
    ----------
    calc: Calculator
        calculator to use 
    elements: list(str)
        list of elements to compute dimer energies for
    plot_file: str / Path
        file to save plot to
    E_range: (float, float) default (-5.0, 5.0)
        max E range to plot
    r_range: (float, float) default (0.5, 10.0)
        range of distances to plot
    """
    Zs = sorted(set([ase.data.chemical_symbols.index(sym) for sym in elements]))
    sym_pairs = [(ase.data.chemical_symbols[Z0], ase.data.chemical_symbols[Z1]) for Z0 in Zs for Z1 in Zs if Z0 <= Z1]

    fig = Figure()
    ax = fig.add_subplot()
    for sym_pair in sym_pairs:
        rs = np.linspace(r_range[0], r_range[1], 100)
        Es = []
        at_0 = Atoms(symbols=sym_pair, positions=[[0, 0, 0], [50.0, 50.0, 50.0]], cell=[100.0] * 3, pbc=[False] * 3)
        at_0.calc = calc
        E_0 = at_0.get_potential_energy()
        for r in rs:
            at = Atoms(symbols=sym_pair, positions=[[0, 0, 0], [r, 0, 0]], cell=[max(r_range) * 2] * 3, pbc=[False] * 3)
            at.calc = calc
            E = at.get_potential_energy()
            if isinstance(calc, BiasCalculator):
                # require unbiased energy if it's defined
                E = at.calc.results_extra["unbiased_energy"]
            Es.append((E - E_0))
        ax.plot(rs, Es, '-', label="-".join(sym_pair))

    ylim = ax.get_ylim()
    if ylim[1] - ylim[0] > max_E_range[1] - max_E_range[0]:
        # clip # range
        ylim = (max(ylim[0], max_E_range[0]), min(ylim[1], max_E_range[1]))
        ax.set_ylim(*ylim)

    fig.legend()
    fig.savefig(plot_file, bbox_inches='tight')
