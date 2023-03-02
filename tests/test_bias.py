import pytest

import numpy as np

from ACEHAL.bias_calc import BiasCalculator, TauRelController


def test_tau_finite_diff_forces(fit_data, fit_model):
    committee_calc = fit_model
    fit_configs, test_configs, _, _, _ = fit_data

    def _do_test(bias_calc, at, label):
        at.calc = committee_calc
        E0_unbiased = at.get_potential_energy()

        at.calc = bias_calc
        E0 = at.get_potential_energy()
        F0 = at.get_forces()
        p0 = at.positions.copy()

        print("FD test E0_unbiased", E0_unbiased, "E0", E0)

        for dx_mag_i in [5]:
            dx_mag = np.sqrt(0.1) ** dx_mag_i

            np.random.seed(10)
            dx = np.random.normal(size=at.positions.shape)
            dx /= np.linalg.norm(dx)

            at.positions = p0 - dx * dx_mag
            Em = at.get_potential_energy()
            at.positions = p0 + dx * dx_mag
            Ep = at.get_potential_energy()

            F_FD = -(Ep - Em) / (2.0 * dx_mag)
            F_analytical = np.sum(F0 * dx)
            print("FD test", label, f"{dx_mag:5.3f}", F_FD, F_analytical, F_FD - F_analytical)

        assert np.abs(F_FD - F_analytical) < 1.0e-5, f"FD failed {F_FD} != {F_analytical} " + label

    for tau in [0.0, 1.0]:
        fit_at = fit_configs[0].copy()
        test_at = test_configs[0].copy()

        bias_calc = BiasCalculator(committee_calc, tau=tau)

        _do_test(bias_calc, fit_at, label=f"fit tau={tau}")
        _do_test(bias_calc, test_at, label=f"test tau={tau}")


def test_tau_finite_diff_stress(fit_data, fit_model):
    committee_calc = fit_model
    fit_configs, test_configs, _, _, _ = fit_data

    def _do_test(bias_calc, at, label):
        at.calc = committee_calc
        E0_unbiased = at.get_potential_energy()

        at.calc = bias_calc
        E0 = at.get_potential_energy()
        V0 = -at.get_volume() * at.get_stress(voigt=False)
        c0 = at.cell.copy()

        print("FD test E0_unbiased", E0_unbiased, "E0", E0)

        for dF_mag_i in [7]:
            dF_mag = np.sqrt(0.1) ** dF_mag_i

            errs = []
            for ii in range(3):
                for jj in range(ii, 3):
                    dF = np.zeros((3, 3))
                    dF[ii, jj] += dF_mag

                    cellm = c0 @ (np.eye(3) - dF)
                    at.set_cell(cellm, True)
                    Em = at.get_potential_energy()

                    cellp = c0 @ (np.eye(3) + dF)
                    at.set_cell(cellp, True)
                    Ep = at.get_potential_energy()

                    V_FD = -(Ep - Em) / (2.0 * dF_mag)
                    V_analytical = V0[ii, jj]
                    ## print("FD test", label, f"{dF_mag:5.3f}", V_FD, V_analytical, V_FD - V_analytical)
                    errs.append(V_FD - V_analytical)
            print("FD test max over components", label, f"{dF_mag:5.3f}", np.max(np.abs(errs)))

        assert np.max(np.abs(errs)) < 2.0e-4

    for tau in [0.0, 1.0]:
        fit_at = fit_configs[0].copy()
        test_at = test_configs[0].copy()

        bias_calc = BiasCalculator(committee_calc, tau=tau)

        _do_test(bias_calc, fit_at, label=f"fit tau={tau}")
        _do_test(bias_calc, test_at, label=f"test tau={tau}")


def test_tau_rel(fit_data, fit_model):
    fit_configs, test_configs, _, _, _ = fit_data

    tau_rel = TauRelController(0.5, 10, delay=3)

    at = fit_configs[0].copy()
    np.random.uniform(10)
    dx = 0.1 * np.random.normal(size=at.positions.shape)
    at.positions += dx

    at.calc = BiasCalculator(fit_model, tau=0.0)
    for i in range(10):
        F = at.get_forces()
        tau_rel.update_calc(at.calc)
        at.positions += 0.01 * F

    assert at.calc.tau == pytest.approx(6.0, abs=3.0)
