import numpy as np

import ase.units
import ase.io


class CellMC:
    """ASE trajectory attachment that does cell MC steps

    Sets CellMC.accept list with # of acceptances, # of trials

    Parameters
    ----------
    atoms: Atoms
        atomic configuration
    temperature_K: float
        temperature in K
    P_GPa: float
        pressure in GPa
    mag: float
        magnitude of cell step perturbations
    """
    def __init__(self, atoms, temperature_K, P_GPa, mag):
        self.atoms = atoms

        self.T = temperature_K
        self.P = P_GPa
        self.P_mag = mag

        self.last_write_step = -1
        self.accept = [0, 0]

    def __call__(self):
        """Do ASE dynamics trajectory attachment action"""

        dF_diag = np.diag(self.P_mag * np.random.normal(size=3))
        # to get comparable magnitude would actually be multiplied by 0.5 here 
        # (since i,j and j,i are both applied), but shear constants are softer than 
        # stretch, going to try bigger steps for off-diagonal
        dF_off_diag = self.P_mag * np.random.normal(size=3)
        dF_off_diag = np.asarray([[0.0, dF_off_diag[0], dF_off_diag[1]],
                                  [dF_off_diag[0], 0.0, dF_off_diag[2]],
                                  [dF_off_diag[1], dF_off_diag[2], 0.0]])
        F = np.eye((3)) + dF_diag + dF_off_diag

        atoms = self.atoms

        orig_cell = atoms.cell.copy()
        E_prev = atoms.get_potential_energy() + atoms.get_volume() * self.P * ase.units.GPa
        atoms.set_cell(orig_cell @ F, True)
        E_new = atoms.get_potential_energy() + atoms.get_volume() * self.P * ase.units.GPa

        self.accept[1] += 1
        if np.random.uniform() < np.exp(-(E_new - E_prev) / (ase.units.kB * self.T)) :
            print(f"Accepted MC cell step from {orig_cell} to {atoms.cell} dE {E_new - E_prev}")
            self.accept[0] += 1
        else:
            # reject
            atoms.set_cell(orig_cell, True)

class SwapMC:
    """ASE trajectory attachment that does swap MC steps

    Parameters
    ----------
    atoms: Atoms
        atomic configuration
    temperature_K: float
        temperature in K
    P_GPa: float
        pressure in GPa
    """
    def __init__(self, atoms, temperature_K, P_GPa):
        self.atoms = atoms

        self.T = temperature_K

        self.last_write_step = -1

    def __call__(self):
        """Do ASE dynamics trajectory attachment action"""

        atoms = self.atoms

        E_prev = atoms.get_potential_energy() 

        els = atoms.get_chemical_symbols()
        ms = atoms.get_masses()
        
        i1 = np.random.randint(len(atoms))
        i_other = np.where(atoms.numbers != atoms.numbers[i1])[0]
        if len(i_other) == 0:
            print("Performing swap step but only single element found!")
        i2 = np.random.choice(i_other)

        m1, m2 = ms[i1], ms[i2]
        el1, el2 = els[i1], els[i2]

        atoms[i1].symbol, atoms[i1].mass = el2, m2
        atoms[i2].symbol, atoms[i2].mass = el1, m1

        E_new = atoms.get_potential_energy() 

        if np.random.uniform() < np.exp(-(E_new - E_prev) / (ase.units.kB * self.T)) :
            print(f"Accepted swap cell step dE {E_new - E_prev}")
            atoms[i1].symbol, atoms[i1].mass = el2, m2
            atoms[i2].symbol, atoms[i2].mass = el1, m1
        else:
            atoms[i1].symbol, atoms[i1].mass = el1, m1
            atoms[i2].symbol, atoms[i2].mass = el2, m2

class HALTolExceeded(Exception):
    pass


class HALMonitor:
    """ASE trajectory attachment that monitors a HAL run for force error exceeding the tolerance
    and also saves the trajectory data and the actual configurations

    Parameters
    ----------
    atoms: Atoms
        atomic configuration
    tol: float
        tolerance for HAL trigger
    tol_eps: float
        regularization for force denominators in tolerance calculator
    tau_rel_control: TauRelController, default None
        object for keeping tau fixed to a fraction of the regular forces
    traj_file: str / Path, default None
        file to save trajectory to
    traj_interval: int, default 10
        interval between saved trajectory snapshots
    err_forces_RMS: bool, default True
        use the RMS force error rather than MAE error
    """
    def __init__(self, atoms, tol, tol_eps, tau_rel_control=None, traj_file=None, traj_interval=10, err_forces_RMS=True):
        self.atoms = atoms
        self.tol = tol
        self.tol_eps = tol_eps
        self.tau_rel_control = tau_rel_control
        self.traj_interval = traj_interval
        self.err_forces_RMS = err_forces_RMS

        if traj_file is not None:
            self.traj_file = open(traj_file, "w")
            self.traj_file_format = ase.io.formats.filetype(self.traj_file, read=False)
        else:
            self.traj_file = None

        self.step = 0
        self.run_data = { 'PE': [], 'KE': [], 'P': [], 'criterion': [] }
        self.HAL_trigger_config = None
        self.HAL_trigger_step = None

        self.last_write_step = None
        self.restart = False


    def write_final_config(self, atoms):
        """Write final config if it hasn't been written before, and close traj file

        Parameters
        ----------
        atoms: Atoms
            final atomic configuration
        """
        if self.traj_file is not None:
            if self.last_write_step != self.step - 1:
                ase.io.write(self.traj_file, atoms, format=self.traj_file_format)
            self.traj_file.close()


    def mark_restart(self):
        """Mark a dynamics restart so that first config isn't saved
        """
        self.restart = True


    def __call__(self):
        """Do ASE dynamics trajectory attachment action"""
        if self.restart:
            # skip first call after restart
            self.restart = False
            return

        atoms = self.atoms

        self.run_data["PE"].append(atoms.get_potential_energy() / len(atoms))
        self.run_data["KE"].append(atoms.get_kinetic_energy() / len(atoms))
        self.run_data["P"].append(-np.mean(atoms.get_stress()[0:3]))

        # # trigger forces so that results_extra["err_forces"] is present, caching should
        # # ensure that forces used for MD do not require an additional calculation
        # _ = atoms.get_forces()

        # check HAL tolerance
        forces_err = atoms.calc.results_extra["err_forces"] if self.err_forces_RMS else atoms.calc.results_extra["err_forces_MAE"]
        # NOTE: add support for softmax
        criterion = np.max(forces_err / (np.linalg.norm(atoms.calc.results_extra["unbiased_forces"], axis=1) + self.tol_eps))
        self.run_data["criterion"].append(criterion)

        atoms.info["HAL_step"] = self.step
        atoms.info["HAL_criterion"] = criterion

        if self.HAL_trigger_config is None and criterion > np.abs(self.tol):
            # first time criterion exceeded tolerance
            self.HAL_trigger_config = atoms.copy()
            self.HAL_trigger_step = self.step
            if self.tol > 0.0:
                # abort dynamics with specific signal
                raise HALTolExceeded

        if self.traj_file is not None and self.step % self.traj_interval == 0:
            # save force errors and criterion in config
            if "HAL_force_err" in atoms.arrays:
                del atoms.arrays["HAL_force_err"]
                atoms.new_array("HAL_force_err", forces_err)
            ase.io.write(self.traj_file, atoms, format=self.traj_file_format)
            self.last_write_step = self.step

        # update tau to maintain tau_rel
        if self.tau_rel_control is not None:
            self.tau_rel_control.update_calc(atoms.calc)

        self.step += 1
