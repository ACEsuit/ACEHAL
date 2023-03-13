from sklearn.linear_model import BayesianRidge
from ase.calculators.castep import Castep
from ase.io import read
from ACEHAL.HAL import HAL

### read inital database 
fit_configs = read("../filename.xyz", ":")

## keys of the DFT labels in the initial database, to be used in HAL configs too, Fmax excludes large forces from fitting
data_keys = { "E" : "energy", "F" : "forces", "V" : "virial", "Fmax" : 15.0 }

## set up (CASTEP) DFT calculator
calculator = Castep()
calculator._directory="./_CASTEP"
calculator.param.cut_off_energy=500
calculator.param.mixing_scheme='Pulay'
calculator.param.write_checkpoint='none'
calculator.param.smearing_width=0.1
calculator.param.finite_basis_corr='automatic'
calculator.param.calculate_stress=True
calculator.param.max_scf_cycles=250
calculator.cell.kpoints_mp_spacing=0.04

## set isolated atom energies or E0s (possible to use numbers from CASTEP pseudopotential file)
E0s = { "Al" : -105.8114973092, "Si" : -163.2225204255 }

## weights, the denominators may be interpreted as GAP sigmas
weights = { "E_per_atom": 1.0 / 0.001, "F": 1.0 / 0.1, "V_per_atom": 1.0 / 0.01 }

## sklearn BRR solver
solver = BayesianRidge(fit_intercept=True, compute_score=True)

## cor_order, r_cut fixed, whereas maxdeg is optimised
fixed_basis_info = {"elements": list(E0s.keys()), "cor_order" : 2, "r_cut" : 5.0,  "smoothness_prior" : None }
optimize_params = {"maxdeg": ("int", (3, 16))}

HAL(fit_configs, # initial fitting database
    fit_configs, # initial starting datbase for HAL (often equal to initial fitting datbase)
    None, # use ACE1x defaults (advised)
    solver, # sklearns solver to be used for fitting 
    fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights}, # fitting arguments
    n_iters=100, # max HAL iterations 
    traj_len=2000, # max steps during HAL iteration until config is evaluated using QM/DFT
    tol=0.3, # relative uncertainty tolerance [0.2-0.4]
    tol_eps=0.2,  # regularising fraction of uncertainty [0.1-0.2]
    tau_rel=0.2, # biasing strength [0.1-0.3], e.g. biasing strength relative to regular MD forces
    ref_calc=calculator, # reference QM/DFT calculator
    dt_fs=1.0, # timestep (in fs)
    T_K=800,  # temperature (in K)
    T_timescale_fs=100, # Langevin thermostat length scale (in fs)
    P_GPa=1.0, # Pressure (in GPa)
    swap_step_interval=50, # atom swap MC step interval 
    cell_step_interval=50, # cell shape MC step interval
    basis_optim_kwargs={"n_trials": 5, # max number of basis optimisation iterations
                        "timeout" : 10000, # timeout for a basis optimisation iteration
                        "max_basis_len": 3000, # max basis size 
                        "fixed_basis_info": fixed_basis_info, # fixed basis information (see above)
                        "optimize_params": optimize_params}, # optimisable parameter (see above)
    basis_optim_interval=1, # interval of basis optimisation  
    file_root="test_HAL", # file root for names
    test_fraction=0.1, # fraction of config sampled for test database
    traj_interval=10) # interval of saving snapshots of trajectory
