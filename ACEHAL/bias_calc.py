import numpy as np

from ase.calculators.calculator import Calculator


class BiasCalculator(Calculator):
    """
    ASE-compatible calculator that produces biased forces given a committee of energies and forces

    Requires calculators that stores at least energy and force committee results in 
    `Calculator.resulte_extra["energy_committee"]`, and analogously for "free_energy" and "forces"

    Parameters
    ----------
    committee_calc: ASE Calculator
        calculator that can compute properties (energy, forces, stresses) and committees of same properties
    tau: float
        magnitude of absolute bias contribution
    """
    implemented_properties = ['forces', 'energy', 'free_energy', 'stress']
    default_parameters = {}
    name = 'BiasCalculator'


    def __init__(self, committee_calc, tau):
        Calculator.__init__(self)

        self.committee_calc = committee_calc
        self.tau = tau

        self.results_extra = {}


    def set_tau(self, tau):
        """Set the value of tau

        Parameters
        ----------
        tau: float
            scale for bias force
        """
        self.tau = tau


    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # if results was wiped, make sure to wipe results_extra as well
        if len(self.results) == 0:
            self.results_extra = {}

        if "forces" or "stress" in properties and "energy" not in properties:
            properties += ["energy"]

        self.committee_calc.calculate(atoms, properties + [p + "_committee" for p in properties], system_changes)
        E = self.committee_calc.results["energy"]
        Es = self.committee_calc.results_extra["energy_committee"]
        # NOTE: original implementation used mean of committee rather than true mean to compute variance, but
        #       this is almost certainly better
        E_bias = np.sqrt(np.mean((Es - E) ** 2))

        if 'energy' in properties or "free_energy" in properties:
            for p in ['energy', 'free_energy']:
                if p in properties:
                    E = self.committee_calc.results[p]
                    self.results[p] = E - self.tau * E_bias

                    self.results_extra["unbiased_" + p] = E
                    self.results_extra["unscaled_bias_" + p] = - E_bias

                    self.results_extra["err_" + p] = self.committee_calc.results_extra["err_" + p]

        if 'forces' in properties:
            F = self.committee_calc.results["forces"]
            Fs = self.committee_calc.results_extra["forces_committee"]

            F_bias = np.mean([(E_comm_member - E) * (F_comm_member - F) for E_comm_member, F_comm_member in zip(Es, Fs)], axis=0)
            F_bias /= E_bias
            self.results["forces"] = F - self.tau * F_bias

            self.results_extra["unbiased_forces"] = F
            self.results_extra["unscaled_bias_forces"] = - F_bias

            self.results_extra["err_forces"] = self.committee_calc.results_extra["err_forces"]
            self.results_extra["err_forces_MAE"] = self.committee_calc.results_extra["err_forces_MAE"]
        if 'stress' in properties:
            S = self.committee_calc.results["stress"]
            Ss = self.committee_calc.results_extra["stress_committee"]

            S_bias = np.mean([(E_comm_member - E) * (S_comm_member - S) for E_comm_member, S_comm_member in zip(Es, Ss)], axis=0)
            S_bias /= E_bias
            self.results["stress"] = S - self.tau * S_bias

            self.results_extra["unbiased_stress"] = S
            self.results_extra["unscaled_bias_stress"] = - S_bias

            self.results_extra["err_stress"] = self.committee_calc.results_extra["err_stress"]


class TauRelController():
    """Calculate an absolute tau for BiasCalculator from a relative tau and running
    averages of unbiased and bias forces

    Sets tau on the calculator whenever updated

    Parameters
    ----------
    tau_rel: float
        scale of bias forces as a fraction of unbiased forces
    tau_hist: int
        length of time over which to smooth force averages
    delay: int, default None
        delay before which starts to be tau adjusted
    no_exp: bool, default False
        don't use exponential smoothing
    """

    def __init__(self, tau_rel, tau_hist, delay=None, no_exp=False):
        self.tau_rel = tau_rel
        self.tau_hist = tau_hist
        if delay is None:
            self.delay = tau_hist
        else:
            self.delay = delay
        self.no_exp = no_exp

        self.mixing = 1.0 / tau_hist
        if self.no_exp:
            self.mean_F_hist = []
            self.bias_F_hist = []
        else:
            self.mean_F = None
            self.bias_F = None
        self.counter = 0


    def update_calc(self, calc):
        """Update internal averages and counters and set new calculator tau

        Parameters
        ----------
        calc: BiasCalculator
            calculator to update
        """
        self.counter += 1

        # NOTE: should this really be mean, or RMS
        mean_F = np.mean(np.linalg.norm(calc.results_extra["unbiased_forces"], axis=1))
        bias_F = np.mean(np.linalg.norm(calc.results_extra["unscaled_bias_forces"], axis=1))
        if self.no_exp:
            # NOTE: original implementation used this no_exp implementation rather than 
            # exponential smoothing
            self.mean_F_hist.append(mean_F)
            self.bias_F_hist.append(bias_F)
            if len(self.mean_F_hist) > self.tau_hist:
                del self.mean_F_hist[0]
                del self.bias_F_hist[0]
            self.mean_F = np.mean(self.mean_F_hist)
            self.bias_F = np.mean(self.bias_F_hist)
        else:
            if self.mean_F is None:
                self.mean_F = mean_F
                self.bias_F = bias_F
            else:
                self.mean_F = (1.0 - self.mixing) * self.mean_F + self.mixing * mean_F
                self.bias_F = (1.0 - self.mixing) * self.bias_F + self.mixing * bias_F

        if self.counter > self.delay:
            tau = self.tau_rel * self.mean_F / self.bias_F
            calc.set_tau(tau)
