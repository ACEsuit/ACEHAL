from sklearn.linear_model import  ARDRegression
from ase.calculators.orca import ORCA
from ase.io import read
from ACEHAL.HAL import HAL

### read inital database 
fit_configs = read("../filename.xyz", ":")

## keys of the DFT labels in the initial database, to be used in HAL configs too, Fmax excludes large forces from fitting
data_keys = { "E" : "energy", "F" : "forces", "V" : "virial", "Fmax" : 15.0 }

## set up (ORCA) DFT calculator
calculator = ORCA(label="orca",
            orca_command='/opt/womble/orca/orca_4_2_1_linux_x86-64_openmpi314/orca',
            charge=0, mult=1, task='gradient',
            orcasimpleinput='RKS wB97X 6-31G(d) Grid6 FinalGrid6 NormalSCF',
            orcablocks="" )

## set isolated atom energies or E0s
E0s = { "H" : -13.587222780835477, "C" : -1029.4889999855063, "O" : -2041.9816003861047 }

## weights, the denominators may be interpreted as GAP sigmas
weights = { "E_per_atom": 1.0 / 0.001, "F": 1.0 / 0.1, "V_per_atom": 1.0 / 0.01 }

## sklearn ARD solver, adjusting threshold_lambda sets sparsity of the fit
solver = ARDRegression(threshold_lambda=10000, fit_intercept=True, compute_score=True)

## no basis optimisation, e.g. fixed basis is constant and defined below 
fixed_basis_info = {"elements": list(E0s.keys()), "cor_order" : 2, "maxdeg" : 12, "r_cut" : 5.0,  "smoothness_prior" : None }

HAL(fit_configs, # initial fitting database
    fit_configs, # initial starting datbase for HAL (often equal to initial fitting datbase)
    None, # use ACE1x defaults (advised)
    solver, # sklearns solver to be used for fitting 
    fit_kwargs={"E0s": E0s, "data_keys": data_keys, "weights": weights}, #fitting arguments
    n_iters=100, # max HAL iterations 
    traj_len=2000, # max steps during HAL iteration until config is evaluated using QM/DFT
    tol=0.3, # relative uncertainty tolerance [0.2-0.4]
    tol_eps=0.2,  # regularising fraction of uncertainty [0.1-0.2]
    tau_rel=0.2, # biasing strength [0.1-0.3], e.g. biasing strength relative to regular MD forces
    ref_calc=calculator, # reference QM/DFT calculator
    dt_fs=1.0, # timestep (in fs)
    T_K=300, # temperature (in K)
    T_timescale_fs=100, # Langevin thermostat length scale (in fs)
    default_basis_info=fixed_basis_info, # set fixed (!) basis info
    file_root="test_HAL", # file root for names
    test_fraction=0.1, # fraction of config sampled for test database
    traj_interval=10) # interval of saving snapshots of trajectory