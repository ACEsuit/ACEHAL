import pytest
from pathlib import Path

from julia import JuliaError

from ACEHAL import basis


def test_basis_default():
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg': 4, 'r_cut': 5.0, 'smoothness_prior': None}
    B, len_B, normalization = basis.define_basis(params)
    assert len_B == 10
    assert normalization is None

def test_basis_smooth():
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg': 4, 'r_cut': 5.0, 'smoothness_prior': ('algebraic', 2)}
    B, len_B, normalization = basis.define_basis(params)
    assert len_B == 10
    assert len(normalization) == len_B

def test_basis_missing_param():
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg': 4, 'r_cut': 5.0}
    with pytest.raises(ValueError):
        B, len_B, normalization = basis.define_basis(params)

def test_basis_extra_param():
    # extra param OK
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg': 4, 'r_cut': 5.0, 'smoothness_prior': None,'r_alt': 4}
    B, len_B, normalization = basis.define_basis(params)

def test_basis_str():
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg_ACE': 4, 'maxdeg_pair': 8, 'r_cut_ACE': 5.0, 'r_cut_pair': 5.0, 'r_in': 2.0, 'r_0': 3.0}
    with open(Path(__file__).parent / "assets" / "basis") as fin:
        julia_source = fin.read()
    B, len_B, normalization = basis.define_basis(params, julia_source=julia_source)
    assert len_B == 14
    assert normalization is None

def test_basis_str_julia_error():
    params = {'elements': ['Si'], 'cor_order': 3, 'maxdeg_ACE': 4, 'maxdeg_pair': 8, 'r_cut_ACE': 5.0, 'r_cut_pair': 5.0, 'r_in': 2.0}
    with open(Path(__file__).parent / "assets" / "basis") as fin:
        julia_source = fin.read()
    with pytest.raises(JuliaError):
        B, len_B, normalization = basis.define_basis(params, julia_source=julia_source)
