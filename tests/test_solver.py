import pytest

import numpy as np

from sklearn.linear_model import BayesianRidge, ARDRegression
from ACEHAL.bayes_regress_max import BayesianRegressionMax

def make_A_rhs(coef, pts, noise):
    A = np.zeros(shape=(len(pts), len(coef)))
    for i in range(len(coef)):
        A[:, i] = pts ** (i + 1)
    rhs = A @ coef + noise * np.random.uniform(size=len(pts))

    return A, rhs

def test_BRM_BRR():
    np.random.seed(5)
    coef = np.random.uniform(size=20)
    pts = np.random.uniform(size=200)

    A, rhs = make_A_rhs(coef, pts, 0.03)

    s = BayesianRidge()
    s.fit(A, rhs)
    print("BayesianRidge")
    print(s.coef_)
    resid = np.linalg.norm(A @ s.coef_ - rhs)
    print("resid", resid, np.linalg.norm(rhs))
    assert resid == pytest.approx(0.217, 1e-2)

    s = BayesianRegressionMax()
    s.fit(A, rhs)
    print("BayesianRegressionMax BRR")
    print(s.coef_)
    resid = np.linalg.norm(A @ s.coef_ - rhs)
    print("resid", resid, np.linalg.norm(rhs))
    assert resid == pytest.approx(0.125, 1e-2)

def test_BRM_ARD():
    np.random.seed(5)
    coef = np.random.uniform(size=20)
    pts = np.random.uniform(size=100)

    A, rhs = make_A_rhs(coef, pts, 0.03)

    s = ARDRegression(threshold_lambda=1000)
    s.fit(A, rhs)
    print("ARDRegression")
    print(s.coef_)
    resid = np.linalg.norm(A @ s.coef_ - rhs)
    print("resid", resid, np.linalg.norm(rhs))
    assert resid == pytest.approx(0.0975, 1e-2)

    s = BayesianRegressionMax(method="ARD")
    s.fit(A, rhs)
    print("BayesianRegressionMax ARD")
    print(s.coef_)
    resid = np.linalg.norm(A @ s.coef_ - rhs)
    print("resid", resid, np.linalg.norm(rhs))
    assert resid == pytest.approx(0.0790, 1e-2)
