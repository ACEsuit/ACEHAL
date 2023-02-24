import importlib

#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

_default_mod = "ACEHAL.bases.default"

def define_basis(basis_info, julia_source=None):
    f"""define an ACE basis using julia

    Runs julia source code that defines B, len_B and P_diag julia variables containing
    the basis, its length, and an optional normalization vector.  Parameters 
    will be passed into julia as a dict named "basis_info".  Julia code must define
    "B" for the basis, "B_len" for its length, and "P_diag" vector of the same
    lenth for an optional basis normalization.

    Parameters
    ----------
    basis_info: dict
        parameters that are used by the julia source to construct the basis
    julia_source: str, default "{_default_mod}"
        Name of julia module defining string "source" with julia source
        and "params" list with required basis_info dict keys, or julia
        source code that defines the required symbols.

    Returns
    -------
    B: julia basis
    B_length int: length of basis
    normalization: numpy array(n_basis) or None normalization needed, e.g. to enforce smoothness prior
    """

    if julia_source is None:
        julia_source = _default_mod

    try:
        basis_mod = importlib.import_module(julia_source)
        julia_source = basis_mod.source
        req_params = set(basis_mod.params)
        if len(req_params - set(basis_info.keys())) > 0:
            raise ValueError(f"Trying to construct julia basis from {basis_mod} "
                             f"with missing required params {req_params - set(basis_info)} and extra params {set(basis_info) - req_params}")
    except ModuleNotFoundError:
        pass

    Main.basis_info = basis_info

    Main.eval(julia_source)

    return Main.B, Main.B_length, Main.P_diag
