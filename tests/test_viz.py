from ACEHAL.viz import error_table

def test_error_table(fit_data, fit_model):
    df = error_table(fit_data[1], fit_model, {"E": "REF_energy", "F": "REF_forces", "V": "REF_virial"})
    print(df.to_string())
