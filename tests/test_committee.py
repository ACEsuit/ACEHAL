import numpy as np

def test_fit_properties(fit_data, fit_model):
    calc = fit_model
    fit_configs, test_configs, _, _, _ = fit_data
    props = ['E', 'F', 'S']

    fit_errs = {p: [] for p in props}
    fit_vars = {p: [] for p in props}
    for at in fit_configs:
        at.calc = calc
        E = at.get_potential_energy() / len(at)
        Es = np.asarray(at.calc.results_extra['energy_committee']) / len(at)
        F = at.get_forces()
        Fs = np.asarray(at.calc.results_extra['forces_committee'])
        S = at.get_stress()
        Ss = np.asarray(at.calc.results_extra['stress_committee'])

        # empirical thresholds. Can probably also use something like 4 * std dev on mean, but still
        # no guarantees it'd work except in the large committee limit
        #
        # note that even with setting seed, fit is not exactly deterministic,
        # and roundoff-level changes in coefficients and coefficient covariance
        # will change committee predictions

        # check correct mean
        fit_errs['E'].append(E - np.mean(Es))
        fit_errs['F'].extend((F - np.mean(Fs, axis=0)).reshape(-1))
        fit_errs['S'].extend((S - np.mean(Ss, axis=0)).reshape(-1))

        fit_vars['E'].extend(Es - E)
        fit_vars['F'].extend(Fs - F)
        fit_vars['S'].extend(Ss - S)

    print("fit E mean err", np.sqrt(np.mean(np.asarray(fit_errs['E']) ** 2)))
    print("fit F mean err", np.sqrt(np.mean(np.asarray(fit_errs['F']) ** 2)))
    print("fit S mean err", np.sqrt(np.mean(np.asarray(fit_errs['S']) ** 2)))

    print("fit E var", np.sqrt(np.mean(np.asarray(fit_vars['E']) ** 2)))
    print("fit F var", np.sqrt(np.mean(np.asarray(fit_vars['F']) ** 2)))
    print("fit S var", np.sqrt(np.mean(np.asarray(fit_vars['S']) ** 2)))

    assert np.sqrt(np.mean(np.asarray(fit_errs['E']) ** 2)) < 6e-5
    assert np.sqrt(np.mean(np.asarray(fit_errs['F']) ** 2)) < 6e-5
    assert np.sqrt(np.mean(np.asarray(fit_errs['S']) ** 2)) < 4e-5

    assert np.sqrt(np.mean(np.asarray(fit_vars['E']) ** 2)) < 3e-4
    assert np.sqrt(np.mean(np.asarray(fit_vars['F']) ** 2)) < 3e-4
    assert np.sqrt(np.mean(np.asarray(fit_vars['S']) ** 2)) < 3e-4

    test_errs = {p: [] for p in props}
    test_vars = {p: [] for p in props}
    for at in test_configs:
        at.calc = calc
        E = at.get_potential_energy()
        Es = np.asarray(at.calc.results_extra['energy_committee'])
        F = at.get_forces()
        Fs = np.asarray(at.calc.results_extra['forces_committee'])
        S = at.get_stress()
        Ss = np.asarray(at.calc.results_extra['stress_committee'])

        # check correct mean
        test_errs['E'].append(E - np.mean(Es))
        test_errs['F'].extend((F - np.mean(Fs, axis=0)).reshape(-1))
        test_errs['S'].extend((S - np.mean(Ss, axis=0)).reshape(-1))

        test_vars['E'].extend(Es - E)
        test_vars['F'].extend(Fs - F)
        test_vars['S'].extend(Ss - S)

    print("test E mean err", np.sqrt(np.mean(np.asarray(test_errs['E']) ** 2)))
    print("test F mean err", np.sqrt(np.mean(np.asarray(test_errs['F']) ** 2)))
    print("test S mean err", np.sqrt(np.mean(np.asarray(test_errs['S']) ** 2)))

    print("test E var", np.sqrt(np.mean(np.asarray(test_vars['E']) ** 2)))
    print("test F var", np.sqrt(np.mean(np.asarray(test_vars['F']) ** 2)))
    print("test S var", np.sqrt(np.mean(np.asarray(test_vars['S']) ** 2)))

    assert np.sqrt(np.mean(np.asarray(test_errs['E']) ** 2)) < 1.5e-2
    assert np.sqrt(np.mean(np.asarray(test_errs['F']) ** 2)) < 1.5e-4
    assert np.sqrt(np.mean(np.asarray(test_errs['S']) ** 2)) < 6.0e-4

    assert np.sqrt(np.mean(np.asarray(test_vars['E']) ** 2)) > 0.01
    assert np.sqrt(np.mean(np.asarray(test_vars['F']) ** 2)) > 0.0001
    assert np.sqrt(np.mean(np.asarray(test_vars['S']) ** 2)) > 0.0003


def test_caching(fit_data, fit_model):
    calc = fit_model
    fit_configs, _, _, _, _ = fit_data

    at = fit_configs[0].copy()
    at.calc = calc

    at.get_potential_energy()
    E = at.calc.results["energy"]
    assert set(at.calc.results.keys()) == set(["energy", "free_energy"])
    assert set(at.calc.results_extra.keys()) == set(["energy_committee", "free_energy_committee", "err_energy", "err_free_energy"])
    at.get_forces()
    F = at.calc.results["forces"][0, 0]
    assert at.calc.results["energy"] == E
    assert set(at.calc.results.keys()) == set(["energy", "free_energy", "forces"])
    assert set(at.calc.results_extra.keys()) == set(["energy_committee", "free_energy_committee", "forces_committee",
                                                     "err_energy", "err_free_energy", "err_forces", "err_forces_MAE"])

    at.positions[0, 0] += 0.1

    at.get_potential_energy()
    Epert = at.calc.results["energy"]
    assert Epert != E
    assert set(at.calc.results.keys()) == set(["energy", "free_energy"])
    assert set(at.calc.results_extra.keys()) == set(["energy_committee", "free_energy_committee", "err_energy", "err_free_energy"])
    at.get_forces()
    Fpert = at.calc.results["forces"][0, 0]
    assert Fpert != F
    assert at.calc.results["energy"] == Epert
    assert set(at.calc.results.keys()) == set(["energy", "free_energy", "forces"])
    assert set(at.calc.results_extra.keys()) == set(["energy_committee", "free_energy_committee", "forces_committee",
                                                     "err_energy", "err_free_energy", "err_forces", "err_forces_MAE"])
