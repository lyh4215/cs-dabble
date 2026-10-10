import numpy as np


layer_a = np.array([
    10.0, 8.0, 6.0, 4.0, 2.0
])

layer_b = np.array([
    1.0, 0.8, 0.6, 0.4, 0.2
])


q = 0.5       # median
fixed = 1.0


for name, spectrum in [
    ("Layer A", layer_a),
    ("Layer B", layer_b),
]:

    quantile_bound = np.quantile(spectrum, q)

    print(name)
    print("spectrum:", spectrum)

    print(
        "fixed bound:",
        fixed
    )

    print(
        "quantile bound:",
        quantile_bound
    )

    print(
        "fixed / median:",
        fixed / np.median(spectrum)
    )

    print(
        "quantile / median:",
        quantile_bound / np.median(spectrum)
    )

    print()