import numpy as np

from ase.constraints import full_3x3_to_voigt_6_stress

from ACEHAL import fit


def test_fit_residuals(fit_data, fit_model_all_info):
    _, Psi, Y, coef, prop_row_inds, n_obs, (B, B_len, B_norm) = fit_model_all_info
    # check accuracy
    fit_residual = np.sqrt(np.mean((Psi @ coef - Y) ** 2))
    for prop in ["E", "F", "V"]:
        print("fit_residual", prop, np.sqrt(np.mean((Psi[prop_row_inds[prop]] @ coef - Y[prop_row_inds[prop]]) ** 2)))
    print("fit_residual", fit_residual)

    # check matrix sizes
    assert Psi.shape == (n_obs, B_len)
    assert Y.shape == (n_obs,)

    # check test residuals
    _, test_configs, E0s, data_keys, weights = fit_data

    test_Psi, test_Y, test_prop_row_inds = fit.assemble_Psi_Y(test_configs, B, E0s,
                                                              data_keys=data_keys, weights=weights)

    test_residual = np.sqrt(np.mean((test_Psi @ coef - test_Y) ** 2))
    for prop in ["E", "F", "V"]:
        print("test_residual", prop, np.sqrt(np.mean((test_Psi[test_prop_row_inds[prop]] @ coef - test_Y[test_prop_row_inds[prop]]) ** 2)))
    print("test_residual", test_residual)

    assert fit_residual < 0.0007
    assert test_residual > 0.0007


def test_fit_properties(fit_data, fit_model):
    calc = fit_model
    fit_configs, test_configs, E0s, data_keys, _ = fit_data
    props = ['E', 'F', 'V']


    fit_diffs = {prop: [] for prop in props}
    for at in fit_configs:
        at.calc = calc
        for prop in props:
            if prop == 'E' and data_keys['E'] in at.info:
                fit_diffs[prop].append((at.get_potential_energy() - at.info[data_keys['E']]) / len(at))
            elif prop == 'F' and data_keys['F'] in at.arrays:
                fit_diffs[prop].extend((at.get_forces() - at.arrays[data_keys['F']]).reshape((-1)))
            elif prop == 'V' and data_keys['V'] in at.info:
                V_voigt = full_3x3_to_voigt_6_stress(at.info[data_keys['V']])
                fit_diffs[prop].extend((-at.get_volume() * at.get_stress() - V_voigt).reshape((-1)) / len(at))

    for p in fit_diffs:
        fit_diffs[p] = np.asarray(fit_diffs[p])

    print("fit RMS Es", np.sqrt(np.mean(fit_diffs['E'] ** 2)))
    print("fit RMS Fs", np.sqrt(np.mean(fit_diffs['F'] ** 2)))
    print("fit RMS Vs", np.sqrt(np.mean(fit_diffs['V'] ** 2)))
    assert np.sqrt(np.mean(fit_diffs['E'] ** 2)) < 0.0001
    assert np.sqrt(np.mean(fit_diffs['F'] ** 2)) < 0.001
    assert np.sqrt(np.mean(fit_diffs['V'] ** 2)) < 0.003

    test_diffs = {prop: [] for prop in props}
    for at in test_configs:
        at.calc = calc
        for prop in props:
            if prop == 'E' and data_keys['E'] in at.info:
                test_diffs[prop].append((at.get_potential_energy() - at.info[data_keys['E']]) / len(at))
            elif prop == 'F' and data_keys['F'] in at.arrays:
                test_diffs[prop].extend((at.get_forces() - at.arrays[data_keys['F']]).reshape((-1)))
            elif prop == 'V' and data_keys['V'] in at.info:
                V_voigt = full_3x3_to_voigt_6_stress(at.info[data_keys['V']])
                test_diffs[prop].extend((-at.get_volume() * at.get_stress() - V_voigt).reshape((-1)) / len(at))

    for p in test_diffs:
        test_diffs[p] = np.asarray(test_diffs[p])

    print("test RMS Es", np.sqrt(np.mean(test_diffs['E'] ** 2)))
    print("test RMS Fs", np.sqrt(np.mean(test_diffs['F'] ** 2)))
    print("test RMS Vs", np.sqrt(np.mean(test_diffs['V'] ** 2)))
    assert np.sqrt(np.mean(test_diffs['E'] ** 2)) > 5e-5
    assert np.sqrt(np.mean(test_diffs['F'] ** 2)) > 0.0002
    assert np.sqrt(np.mean(test_diffs['V'] ** 2)) > 0.002
