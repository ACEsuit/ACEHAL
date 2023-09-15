# Author: Noam Bernstein, U S Naval Research Laboratory
# 
# python re-implementation of julia version from github ACEsuite/ACEfit.jl repo

import sys
import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.optimize import minimize

from sklearn.linear_model._base import LinearModel
from sklearn.base import RegressorMixin

class PrepOnly(Exception):
    pass

class BayesianRegressionMax(RegressorMixin, LinearModel):
    """Bayesian Ridge Regression (BRR or BLR) or Automatic Relevance Determination (ARD) 
    regression by maximizing the log likelihood

    Parameters
    ----------
    method: "BRR" or "BRR_SVD", "ARD", default "BRR"
        regression method to use
    n_iter: int, default 1000
        Maximum number of likelihood maximization iterations
    tol: float, 1.0e-6
        default convergence tolerance (method dependent)
    ftol: float, default None
        convergence tolerance on log likelihood (only supported by some optim_method)
    gtol: float, default None
        convergence tolerance on log likelihood gradient (only supported by some optim_method)
    xtol: float, default None
        convergence tolerance on coefficients (only supported by some optim_method)
    var_c_min: float, default 1e-6
        minimum variance for coefficients
    var_e_min: float, default 1e-6
        minimum noise variance for observations
    var_c_0: float, default 0.1
        initial guess for variance for coefficients
    var_e_0: float, default 0.1
        initial guess for noise variance for observations
    threshold: float, default 10 for ARD only
        threshold for ARD pruning coefficients (keep those with larger values of variance), expressed as 
        factor relative to var_c_min
    optim_method: str, default "BFGS"
        method for scipy.optimize.minimize
    transformation: str, default "square"
        function relating optimization parameters and variances
    options: dict
        options to scipy.optimize.minimize
    verbose: bool, default False
        verbose output

    Attributes
    ----------
    coef_: array-like of shape(n_features,)
        Coefficients of the regression model (mean of distribution)
    var_c_: array-like of shape(n_features,)
        Estimated variance of coefficients
    var_e_: float
        estimated variance of observations
    sigma_: array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights
    scores_: array-like
        list of score at each iteration
    """
    def __init__(self, *, method="BRR", n_iter=1000, tol=1.0e-6, ftol=None, gtol=None, xtol=None,
                 var_c_min=1e-6, var_e_min=1e-6, var_c_0=0.1, var_e_0=0.1,
                 threshold=None, optim_method="BFGS", transformation="square",
                 options={}, verbose=False, ard_conv_plot=None):
        assert method in ["BRR", "BRR_SVD", "ARD"]
        self.method = method
        self.n_iter = n_iter
        self.tol = tol
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.var_c_min = var_c_min
        self.var_e_min = var_e_min
        self.var_c_0 = var_c_0
        self.var_c_0_mag = var_c_0
        self.var_e_0 = var_e_0
        self.ard_tol = threshold
        if self.method == "ARD":
            if self.ard_tol is None:
                # default value for ARD
                self.ard_tol = 10.0
        else:
            # not ARD, only None is meaningful
            if self.ard_tol is not None:
                warnings.warn(f"Got method {self.method} and ard_tol {self.ard_tol} not None, value will be ignored")
        self.optim_method = optim_method
        self.verbose = verbose
        self.transformation = transformation
        self.options = options.copy()
        self._result_x = None

        # basic quantities
        self.y = None
        self.X = None

        # derived matrices
        self.XTX = None
        self.S = None
        self.UT_Y = None
        self.V = None

        # ftol is unreliable for L-BFGS-B
        if self.optim_method.lower() == "l-bfgs-b" and self.ftol is None:
            self.ftol = 0.0

        self.ard_conv_plot = ard_conv_plot

    def reset_threshold(self, threshold):
        """save threshold value and redo final solve and filtering of coefficients

        Parameters
        ----------
        threshold: float
            threshold for pruning coefficients relative to variance var_c_min

        Returns
        -------
        self BayesianRegressionMax: solver object with new attributes such as threshold,
            coefficients, sigma etc.
        """

        assert self._result_x is not None

        self.threshold = threshold

        if self.threshold is not None and self.method != "ARD":
            raise RuntimeError(f"Threshold {self.threshold} is not None, only allowed for 'ARD' != {self.method}")

        coefficients, mask, sigma_masked, var_c_out, var_e_out = self._coef_sigma_masked(self._result_x, threshold)

        if self.verbose:
            print("final variance var_e", var_e_out)
            if self.method in ["BRR", "BRR_SVD"]:
                print("final variance var_c", var_c_out)
                for c_el in zip(coefficients):
                    print("final coeff", c_el)
            else:
                for c_el, var_c_el in zip(coefficients, var_c_out):
                        print("final coeff var_c", c_el, var_c_el)

        self.coef_ = coefficients.copy()
        self.var_c_ = var_c_out.copy()
        self.var_e_ = var_e_out
        self.sigma_ = sigma_masked.copy()
        self.mask_ = mask.copy()

        self.alpha_ = 1.0 / self.var_e_
        self.lambda_ = 1.0 / self.var_c_

        return self


    def fit(self, X, y, prep_only=False):
        """Fit the BayesianRegressionMax model to the data

        Parameters
        ----------
        X: float array (n_obervations, n_features)
            design matrix n_obervations x n_features
        y: float vector (n_features,)
            right hand side

        Returns
        -------
        self BayesianRegressionMax
        """
        self.y = y
        self.X = X

        self._result_x, n_iter, converged, log_likelihoods = self._do_fit(prep_only=prep_only)

        self.scores_ = log_likelihoods.copy()

        if self.method == "BRR_SVD":
            var_c, var_e = self._var_c_e(self._result_x)

            len_S = self.S.shape[0]

            self.UT_Y[:len_S] *= var_c * self.S[:len_S] / (var_c * self.S[:len_S] ** 2 + var_e)

            self.coef_ = self.V @ self.UT_Y[:len_S]
            self.var_c_ = var_c
            self.var_e_ = var_e
            self.sigma_ = np.diag(1.0 / (self.S ** 2 / var_e + 1.0 / var_c))

            self.alpha_ = 1.0 / self.var_e_
            self.lambda_ = 1.0 / self.var_c_
        else:
            self.reset_threshold(self.ard_tol)

        return self


    def _var_c_e(self, x):
        """Compute the variances from scipy.optimize.minimize vector

        Parameters
        ----------
        x: float vector
            scipy.optimize.mimize argument vector

        Returns
        -------
        var_c_vec: float vector coefficient variances
        var_e: float observation variance
        """
        if self.method in ["BRR", "BRR_SVD"]:
            assert len(x) == 2
            x_c = x[0]
        else:
            x_c = x[:-1]

        if self.transformation == "none":
            return self.var_c_min + x_c, self.var_e_min + x[-1]
        elif self.transformation == "square":
            return self.var_c_min + x_c ** 2, self.var_e_min + x[-1] ** 2
        else:
            raise ValueError(f"Unknown transformation {transformation}")


    def _var_c_e_grad(self, x, g):
        """Compute the gradient of the minimized function with respect to scipy.optimize.minimize
        arguments based on gradient w.r.t. variances

        Parameters
        ----------
        x: float vector
            scipy.optimize.minimize argument vector
        g: float vector
            gradient w.r.t. variances

        Returns
        -------
        precond_g: gradient of minimize function
        """
        if self.transformation == "none":
            return g
        elif self.transformation == "square":
            return 2.0 * x * g
        else:
            raise ValueError(f"Unknown transformation {transformation}")


    def _mask(self, x, ard_tol=None):
        """Compute mask for which coefficient components are above threshold

        Parameters
        ----------
        x: float vector
            scipy.optimize.mimize argument vector
        ard_tol: float, default None
            optional tolerance overriding self.ard_tol

        Returns
        -------
        mask: bool vector keep
        """
        if self.method in ["BRR", "BRR_SVD"]:
            assert len(x) == 2
            return [True] * self.X.shape[1]
        else:
            if ard_tol is None:
                ard_tol = self.ard_tol
            return self._var_c_e(x)[0] > ard_tol * self.var_c_min


    def _coef_sigma_masked(self, x, ard_tol=None):
        """calculate coefficients of solution after masking ones that are below the tolerance

        Parameters
        ----------
        x: float vector
            variances (vector of coefficients and scalar for observations) outputed by ARD optimization
        ard_tol: float, default None
            optional tolerance overriding self.ard_tol

        Returns
        -------
        coefficients: float vector (X.shape[1]) solution
        mask: bool vector of non-negligible coefficients
        sigma_masked: float array (sum(mask), sum(mask)) covariance of coefficients
        var_c_vec: float vector variance in coefficients
        var_e: float variance of model error
        """

        var_c, var_e = self._var_c_e(x)

        mask = self._mask(x, ard_tol)
        if self.method in ["BRR", "BRR_SVD"]:
            assert len(x) == 2
            masked_var_c = var_c
        else:
            masked_var_c = var_c[mask]

        coefficients_masked, sigma_masked = BayesianRegressionMax._solve(self.y, self.X[:, mask], masked_var_c, var_e)
        coefficients = np.zeros(self.X.shape[1])
        coefficients[mask] = coefficients_masked

        return coefficients, mask, sigma_masked, var_c, var_e


    # staticmethod so it can be applied to a subset of features
    @staticmethod
    def _solve(y, X, var_c, var_e):
        """solve the linear problem to get model coefficients

        Parameters
        ----------
        y: float vector
            RHS vector
        X: float array
            design matrix
        var_c: float / float vector
            variance of coefficients
        var_e: float
            variance of observations

        Returns
        -------
        coefficients: vector solution to linear problem
        sigma: covariance of solution vector
        """
        M = X.shape[1]

        sigma_c_inv = X.T @ X

        sigma_c_inv /= var_e
        if isinstance(var_c, float):
            var_c_vec = var_c * np.ones(M)
        else:
            var_c_vec = var_c
        sigma_c_inv += np.diag(1.0 / var_c_vec)

        C = cho_factor(sigma_c_inv)
        return (1.0 / var_e) * cho_solve(C, X.T @ y), cho_solve(C, np.eye(M))


    @staticmethod
    def _cho_logdet(C):
        """log of determinant of a Cholesky factor (e.g. as returned by scipy.linalg.cho_factor)

        Parameters
        ----------
        C: tuple(float array, bool)
            Triangular (not checked) factor matrix M

        Returns
        -------
        float log(det(M)) = 2 log (det(C))
        """
        if not isinstance(C, tuple) or not isinstance(C[1], bool):
            raise ValueError("_cho_logdet expected a Cholesky factorized matrix as returned by scipy.linalg.cho_factor")
        return 2.0 * np.sum(np.log(np.diag(C[0])))


    def _log_marginal_likelihood_overdetermined(self, var_c, var_e):
        """log marginal likelihood of solution

        Parameters
        ----------
        var_c: float / float vector
            regularization of coefficients
        var_e: float
            regularization of fit values

        Returns
        -------
        lml: float
            log marginal likelihood
        grad: float vector
            gradients w.r.t. var_c and var_e
        """
        X = self.X
        y = self.y
        XTX = self.XTX

        N = X.shape[0]
        M = X.shape[1]

        if XTX is None:
            sigma_c_inv = X.T @ X
        else:
            sigma_c_inv = XTX.copy()
        sigma_c_inv /= var_e
        if isinstance(var_c, float):
            var_c_vec = var_c * np.ones(M)
        else:
            var_c_vec = var_c
        sigma_c_inv += np.diag(1.0 / var_c_vec)

        # Cholesky
        try:
            C = cho_factor(sigma_c_inv)
        except np.linalg.LinAlgError:
            # return a very bad likelihood so hopefully optimizer will back off
            warnings.warn("cho_solve failed when computing log marginal likelihood, returning a very negative value")
            return -1.0e38, 1.0e38 * np.ones(len(var_c_vec) + 1)

        sigma_c = cho_solve(C, np.eye(M))
        mu_c = 1.0 / var_e * cho_solve(C, X.T @ y)
        sigma_c_inv_logdet = BayesianRegressionMax._cho_logdet(C) 

        lml = -0.5 * (sigma_c_inv_logdet + np.sum(np.log(var_c_vec)) + N * np.log(var_e) + N * np.log(2.0 * np.pi))
        X_mu_c = X @ mu_c
        lml -= 0.5 / var_e * y @ (y - X_mu_c)

        grad = 0.5 * (mu_c ** 2 + np.diag(sigma_c) - var_c_vec) / (var_c_vec ** 2)
        if self.verbose:
            grad_0_i = np.where(np.abs(grad) < 1.0e-11)[0]
            print("lml # grad < 1e-11", sum(grad_0_i))
            print("grad_0_i", grad_0_i)
            print("grad[grad_0_i]", grad[grad_0_i])
            print("mu_c[grad_0_i]", mu_c[grad_0_i])
            print("np.diag(sigma_c)[grad_0_i]", np.diag(sigma_c)[grad_0_i])
            print("var_c_vec[grad_0_i]", var_c_vec[grad_0_i])
            print("np.diag(sigma_c_inv)[grad_0_i]", np.diag(sigma_c_inv)[grad_0_i])
            print("sigma_c_inv row norms", np.linalg.norm(sigma_c_inv[grad_0_i][:, grad_0_i], axis=1))
            print("sigma_c row norms", np.linalg.norm(sigma_c[grad_0_i][:, grad_0_i], axis=1))
            print("XTX norms", np.linalg.norm(XTX[grad_0_i][:, grad_0_i], axis=1))
            print("X.T @ X row norms", np.linalg.norm((self.X.T @ self.X)[grad_0_i][:, grad_0_i], axis=1))
            print("X col norms", np.linalg.norm(self.X[:, grad_0_i], axis=0))
        grad = np.append(grad, [0.5 / (var_e ** 2) * (np.sum((y - X_mu_c) ** 2) + np.sum(XTX * sigma_c) - N * var_e)])

        if isinstance(var_c, float):
            return lml, np.asarray([np.sum(grad[:-1]), grad[-1]])
        elif self.method == "ARD":
            return lml, grad
        else:
            raise ValueError


    def _log_marginal_likelihood_underdetermined(self, var_c, var_e):
        """log marginal likelihood of solution

        Parameters
        ----------
        var_c: float / float vector
            regularization of coefficients
        var_e: float
            regularization of fit values

        Returns
        -------
        lml: float
            log marginal likelihood
        grad: float vector
            gradients w.r.t. var_c and var_e
        """
        X = self.X
        y = self.y

        N = X.shape[0]
        M = X.shape[1]

        if isinstance(var_c, float):
            var_c_vec = var_c * np.ones(M)
        else:
            var_c_vec = var_c

        # X @ np.diag(var_c_vec) @ X.T
        sigma_y = (X * var_c_vec) @ X.T
        sigma_y += var_e * np.eye(N)

        C = cho_factor(sigma_y)
        inv_sigma_y_y = cho_solve(C, y)

        # factor of 2 in logdet(C) because C is actually the Cholesky factor
        lml = -0.5 * (y @ inv_sigma_y_y + BayesianRegressionMax._cho_logdet(C) + N * np.log(2 * np.pi))
        grad = 0.5 * (X.T @ inv_sigma_y_y) ** 2
        W = cho_solve(C, X)
        grad -= 0.5 * np.sum(X * W, axis=0)
        grad = np.append(grad, [0.5 * np.sum(inv_sigma_y_y ** 2) - 0.5 * np.trace(cho_solve(C, np.eye(N)))])

        if isinstance(var_c, float):
            return lml, np.asarray([np.sum(grad[:-1]), grad[-1]])
        else:
            return lml, grad


    def _log_marginal_likelihood_svd(self, var_c, var_e):
        """compute BRR log marginal likelihood using svd

        Parameters
        ----------
        var_c: float
            regularization of coefficients
        var_e: float
            regularization of fit values

        Returns
        -------
        lml: float
            log marginal likelihood
        grad: float vector
            gradients w.r.t. var_c and var_e
        """
        assert isinstance(var_c, float)


        N = self.X.shape[0]
        S_dim = self.S.shape[0]

        t = var_c * self.S ** 2 + var_e
        lml = -0.5 * np.sum(self.UT_Y[:S_dim] ** 2 / t + np.log(t))
        dlml_dc = 0.5 * np.sum((self.UT_Y[:S_dim] * self.S / t) ** 2 - self.S ** 2 / t)
        dlml_de = 0.5 * np.sum((self.UT_Y[:S_dim] / t) ** 2 - 1.0 / t)

        if S_dim < N:
            lml += -0.5 * np.sum(self.UT_Y[S_dim:] ** 2) / var_e
            dlml_de += 0.5 * np.sum(self.UT_Y[S_dim:] ** 2) / var_e ** 2

        lml += -0.5 * (N - S_dim) * np.log(var_e)
        dlml_de += -0.5 * (N - S_dim) / var_e
        lml += -0.5 * N * np.log(2.0 * np.pi)

        return lml, np.asarray([dlml_dc, dlml_de])


    def grad_test(self, x=None, dir_vec=None):
        """gradient test comparing analytical to numerical derivative

        Parameters
        ----------
        x: array(float)
            value at which to compute gradient
        dir_vec: array(float)
            direction along which to project gradient

        Returns
        -------
        analytical_deriv, numerical_deriv (float, float) values of projected derivatives
        """
        # copied from _do_fit
        def func_and_grad(x):
            (var_c_use, var_e_use) = self._var_c_e(x)
            val_grad = self._lml_grad(var_c_use, var_e_use)

            val_grad = (-val_grad[0], -self._var_c_e_grad(x, val_grad[1]))

            return val_grad

        if x is None:
            x_0 = self._x.copy()
        else:
            x_0 = x.copy()

        val_0, grad_0 = func_and_grad(x_0)

        if dir_vec is None:
            dir_vec = grad_0
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        for dp_mag_i in range(0, -10, -1):
            dp_mag = np.sqrt(10) ** dp_mag_i
            dp = dir_vec * dp_mag

            xp = x_0 + dp
            val_p, _ = func_and_grad(xp)
            xm = x_0 - dp
            val_m, _ = func_and_grad(xm)

            print("TEST ANALYTICAL", np.linalg.norm(grad_0), "FD", (val_p - val_m) / (2.0 * dp_mag),
                  "DIFF", (val_p - val_m) / (2.0 * dp_mag) - np.linalg.norm(grad_0))

        return self._var_c_e(x_0)[0], grad_0


    def _do_fit(self, prep_only=False):
        """Fit using BRR, BRR_SVD, or ARD

        Returns
        -------
        coefficients: float vector (X.shape[1]) of coefficients
        n_iter: int number of iterations
        converged: bool whether parameter optimization converged
        log_likelihoods: float vectors of scores at each iteration
        """

        # reset matrices that are initializer here, in case they are left over from previous call
        self.XTX = None
        self.S = None
        self.UT_Y = None
        self.V = None

        if self.method in ["BRR", "ARD"]:
            # regular BRR/ARD, different functions for over vs. underdetermined
            if self.X.shape[0] >= self.X.shape[1]:
                self.XTX = self.X.T @ self.X
                self._lml_grad = self._log_marginal_likelihood_overdetermined
            else:
                self._lml_grad = self._log_marginal_likelihood_underdetermined
        elif self.method in ["BRR_SVD"]:
            # BRR With SVD, entirely different calculation
            U, self.S, Vh = svd(self.X, full_matrices=True)
            self.V = Vh.T
            self.UT_Y = U.T @ self.y
            self._lml_grad = self._log_marginal_likelihood_svd

        # kinda horrible closure hack to keep track of function values etc during iteration
        last_iter_val = None
        last_iter_grad_norm = None
        def func_and_grad(x):
            (var_c_use, var_e_use) = self._var_c_e(x)
            val_grad = self._lml_grad(var_c_use, var_e_use)

            val_grad = (-val_grad[0], -self._var_c_e_grad(x, val_grad[1]))

            # set these so _ValTracker can access them
            nonlocal last_iter_val, last_iter_grad_norm

            last_iter_val = -val_grad[0]
            last_iter_grad_norm = np.linalg.norm(val_grad[1])

            if self.verbose:
                print("func_and_grad", last_iter_val, last_iter_grad_norm, "var_e", var_e_use); sys.stdout.flush()

            return val_grad

        # make into a vector of correct length
        if self.method == "ARD":
            self.var_c_0 = self.var_c_0_mag * np.ones(self.X.shape[1])

        # set bounds and initial guesses for all x components
        if self.transformation == "none":
            if self.method in ["BRR", "BRR_SVD"]:
                # one for var_c and one for var_e
                bounds = [(0, None)] * 2
            else:
                # one for each element of var_c vec and one for var_e
                bounds = [(0, None)] * (self.X.shape[1] + 1)
            x_c_0 = [self.var_c_0]
            x_e_0 = self.var_e_0
        elif self.transformation == "square":
            bounds = None
            x_c_0 = self.var_c_0 ** 0.5
            x_e_0 = self.var_e_0 ** 0.5
        else:
            raise ValueError(f"Unknown transformation {transformation}")

        # one var_c per coefficient + var_e
        x0 = np.append(x_c_0, [x_e_0])

        options = {"maxiter": self.n_iter, "disp": self.verbose}
        if self.ftol is not None:
            options["ftol"] = self.ftol
        if self.gtol is not None:
            options["gtol"] = self.gtol
        if self.xtol is not None:
            options["xtol"] = self.xtol
        options.update(self.options)

        class _FtolConv(Exception):
            pass

        class _ValTracker:
            def __init__(self, ftol=None, ftol_interval=10, ard_tol=None, _mask=None, verbose=False,
                         ard_conv_plot=None, _var_c_e=None):
                self.ftol = ftol
                self.ftol_interval = 10
                self.ard_tol = ard_tol
                self._mask = _mask
                self.verbose = verbose
                self.ard_conv_plot = ard_conv_plot
                self._var_c_e = _var_c_e

                self.iter = 0
                self.func_vals = []
                # self.times = []
                # self.func_grad_norms = []
                self.tstart = time.time()

            def __call__(self, x):
                # implicitly get values set in func_and_grad()
                nonlocal last_iter_val, last_iter_grad_norm
                tnow = time.time()
                self.func_vals.append(last_iter_val)
                # self.times.append(tnow - self.tstart)
                # self.func_grad_norms.append(last_iter_grad_norm)
                self.x = x.copy()

                if self.ftol is not None and len(self.func_vals) > self.ftol_interval:
                    val_slope = (self.func_vals[-1] - self.func_vals[-1 - self.ftol_interval]) / self.ftol_interval
                    if val_slope / np.abs(self.func_vals[-1]) < self.ftol:
                        raise _FtolConv

                self.iter += 1
                if self.verbose:
                    out = f"{self.iter} val {last_iter_val:.3f} |grad| {last_iter_grad_norm:.3f}"
                    if self.ard_tol is not None:
                        out += f" # above threshold {sum(self._mask(x))}"
                    print(out + f" time {tnow - self.tstart:.2f} s")
                    sys.stdout.flush()

                if self.ard_conv_plot is not None:
                    self.ard_conv_plot.iteration(1.0 / self._var_c_e(x)[0])

        val_tracker = _ValTracker(ftol=self.ftol, ard_tol=self.ard_tol, _mask=self._mask,
                                  verbose=self.verbose, ard_conv_plot=self.ard_conv_plot, _var_c_e=self._var_c_e)

        self._x = x0
        if prep_only:
            raise PrepOnly

        if self.ard_conv_plot is not None:
            self.ard_conv_plot.iteration(1.0 / self._var_c_e(x0)[0])

        try:
            # Can end up raising numpy.linalg.LinAlgError for bad initial choices of x0, e.g. too large
            # We could perhaps catch and try different values
            result = minimize(func_and_grad, x0, method=self.optim_method, jac=True, tol=self.tol,
                              options=options, callback=val_tracker, bounds=bounds)
            result_x = result.x
            result_nit = result.nit
            result_success = result.success
        except _FtolConv:
            result_x = val_tracker.x
            result_nit = len(val_tracker.func_vals)
            result_success = True

        self._x = result_x

        return result_x, result_nit, result_success, val_tracker.func_vals
