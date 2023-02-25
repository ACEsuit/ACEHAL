import warnings

import numpy as np

import ase.data
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

from julia.JuLIP import energy, forces, stress

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

Main.eval(""" 
using LinearAlgebra

function do_GC()
    GC.gc()
end

function get_com_energies(CO_IP, at)
    E_bar, E_comms = ACE1.co_energy(CO_IP, at)
    return E_comms
end

function get_com_forces(CO_IP, at)
    F_bar, F_comms = ACE1.co_forces(CO_IP, at)
    return F_comms
end

function get_com_virials(CO_IP, at)
    s_bar, s_comms = ACE1.co_virial(CO_IP, at)
    return s_comms
end
""")
from julia.Main import do_GC, get_com_energies, get_com_forces, get_com_virials


class ACECommittee(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy as well as optional
    committee values:w

    After Atoms.get_potential_energy() Calculator.results_extra["energy_committee"] will have an
    array (n_committee) of energies from the committee, and similarly for Atoms.get_forces()
    and "forces_committee" (n_committee, n_atoms, 3) and Atoms.get_stress and "stress_committee" (n_committee, 6).

    Parameters
    ----------
    mean_julip_calc: str
        name of symbol defined in julia.Main containing mean ACE1 calculator
    committee_julip_calc: str, default None
        name of symbol defined in julia.Main containing committee of ACE1 calculators
    GC_interval: int, default 10000
        interval between calls to julia GC.gc()
    """
    implemented_properties = ['forces', 'energy', 'free_energy', 'stress']
    default_parameters = {}
    name = 'ACECommittee'

    n_since_GC = 0


    def __init__(self, mean_julip_calc, committee_julip_calc=None, GC_interval=10000):
        Calculator.__init__(self)

        self.mean_julip_calc = Main.eval(mean_julip_calc)

        if committee_julip_calc is not None:
            self.committee_julip_calc = Main.eval(committee_julip_calc)
            self.results_extra = {}
        else:
            self.committee_julip_calc = None
            self.results_extra = None

        self.GC_interval = 10000


    def __del__(self):
        do_GC()


    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        julia_atoms = convert(ASEAtoms(atoms))

        # if results was wiped, make sure to wipe results_extra as well
        if len(self.results) == 0 and self.committee_julip_calc is not None:
            self.results_extra = {}

        if 'energy' in properties or 'free_energy' in properties:
            E = energy(self.mean_julip_calc, julia_atoms)
            self.results['energy'] = E
            self.results['free_energy'] = E

            if self.committee_julip_calc is not None:
                Es = np.asarray(get_com_energies(self.committee_julip_calc, julia_atoms) )
                self.results_extra['energy_committee'] = Es
                self.results_extra['free_energy_committee'] = Es

                # energy errors from root variance 
                self.results_extra['err_energy'] = np.sqrt(np.mean((Es - E) ** 2))
                self.results_extra['err_free_energy'] = self.results_extra['err_energy']
        if 'forces' in properties:
            F = np.array(forces(self.mean_julip_calc, julia_atoms))
            self.results['forces'] = F

            if self.committee_julip_calc is not None:
                Fs = np.asarray(get_com_forces(self.committee_julip_calc, julia_atoms))
                self.results_extra['forces_committee'] = Fs

                # default force error is root variance (summed over vector components)
                self.results_extra["err_forces"] = np.sqrt(np.mean(np.linalg.norm(Fs - F, axis=2) ** 2, axis=0))
                # similar, but mean of norm over committee, more like mean absolute energy (MAE)
                self.results_extra["err_forces_MAE"] = np.mean(np.linalg.norm(Fs - F, axis=2), axis=0)
        if 'stress' in properties:
            S = full_3x3_to_voigt_6_stress(np.array(stress(self.mean_julip_calc, julia_atoms)))
            self.results['stress'] = S

            if self.committee_julip_calc is not None:
                Vs = get_com_virials(self.committee_julip_calc, julia_atoms)
                vol = atoms.get_volume()
                Ss = np.asarray([-full_3x3_to_voigt_6_stress(V) / vol for V in Vs])
                self.results_extra['stress_committee'] = Ss

                # root of variance of stress components
                self.results_extra["err_stress"] = np.sqrt(np.mean((Ss - S) ** 2))

        if ACECommittee.n_since_GC % self.GC_interval == self.GC_interval-1:
            warnings.warn("Calling julia GC.gc()")
            do_GC()
        ACECommittee.n_since_GC += 1
