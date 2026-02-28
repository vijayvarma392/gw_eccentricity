"""Rational fit."""
import warnings
import numpy as np


def _arnoldi_basis(x, degree, w=None):
    """Arnoldi basis on nodes x upto degree.
    
    Typical polynomial basis (Vandermonde): {1, x, x^2, ..., x^m} is not 
    suitable for rational functions with higher degrees. Instead, we use the 
    Arnoldi process to generate an orthonormal basis 
    {v_0(x), v_1(x), ..., v_m(x)} that is better suited for rational
    approximation.

    parameters:
    -----------
    x: array-like, shape (n,)
        The nodes to build the basis on.
    degree: int
        The degree of basis space.
    w: array-like, shape (n,), default=None
        Initial vector for the Arnoldi process. 
        If None, it will be initialized as a vector of ones.

    returns:
    --------
    V: (n, degree+1) array
        The evaluated Arnoldi basis functions at the nodes x.
    H: (degree+1, degree) array
        The upper Hessenberg matrix from the Arnoldi process, which can be used
        to evaluate the basis at new points without re-running the Arnoldi process.
    v0_norm: float
        The norm of the initial vector w used in the Arnoldi process, which is
        needed to evaluate the first basis function at new points.
    """
    n = len(x)
    V = np.zeros((n, degree + 1))
    H = np.zeros((degree + 1, degree))

    # Start with the first basis function v_0(x) = 1
    V[:, 0] = 1.0 if w is None else w
    v0_norm = np.linalg.norm(V[:, 0])
    V[:, 0] /= v0_norm  # Normalize the first basis function

    for k in range(degree):
        # Compute the next basis function
        v = V[:, k] * x  # Multiply the current basis by x

        # Orthogonalize against previous basis functions
        for j in range(k + 1):
            H[j, k] = np.dot(V[:, j], v)  # Projection onto v_j
            v -= H[j, k] * V[:, j]  # Remove the component in the direction of v_j

        H[k + 1, k] = np.linalg.norm(v)  # Norm of the new basis function
        if H[k + 1, k] > 1e-14:  # Avoid division by zero
            V[:, k + 1] = v / H[k + 1, k]  # Normalize to get v_{k+1}
        else:
            break  # If the new basis function is negligible, stop

    return V, H, v0_norm


def _eval_arnoldi_basis(x, H, degree, v0_norm):
    """Evaluate Arnoldi orthonormal polynomial basis at new points using 
    stored Hessenberg matrix H.

    parameters:
    -----------
    x: 1d array-like
        The new input points to evaluate the basis functions at.
    H: (degree+1, degree) array
        The Hessenberg matrix from the Arnoldi process inside fit() method.
    degree: int
        The degree of the basis to evaluate.
    v0_norm: float
        The norm of the initial weight vector w from the Arnoldi
        process when building the basis at the training points, used
        to compute the first basis function.

    returns:
    --------
    V: (len(x), degree+1) array
        The evaluated basis functions at the new points x.
    """
    x = np.atleast_1d(x)
    n = len(x)
    # Allocate basis matrix
    V = np.zeros((n, degree + 1))
    # Start with the first basis function v_0(x) = 1
    V[:, 0] = 1.0 / v0_norm  # Scale the first basis function using v0_norm

    # Build higher degree basis functions using the stored Hessenberg
    # matrix H
    for k in range(degree):
        v = V[:, k] * x
        v = V[:, k] * x - V[:, :k+1] @ H[:k+1, k]
        if H[k + 1, k] < 1e-14: # If the new basis function is negligible, stop
            break 
        V[:, k + 1] = v / H[k + 1, k]
    return V


def _scale_x(x, x_min, x_max):
    """Scale x for numerical stability of the fit."""
    return (x - x_min) / (x_max - x_min) * 2 - 1


def is_underdetermined(degrees, n_points):
    """Check if the system is underdetermined."""
    return degrees[0] + degrees[1] + 1 >= n_points


def suggest_degrees(degrees, n_points):
    """Suggest best degrees for rational fit given the number of data points.
    
    If the system is underdetermined (m + n + 1 >= n_points), iteratively 
    lower degrees by 1, alternating between denominator first, then numerator,
    until the system is well-determined.

    parameters:
    -----------
    degrees: (int, int)
        The initial (numerator_degree, denominator_degree).
    n_points: int
        The number of data points.

    returns:
    --------
    (int, int)
        The suggested (numerator_degree, denominator_degree).
    """
    m, n = degrees
    reduce_denom = True  # Start by reducing denominator

    while is_underdetermined((m, n), n_points):
        if m <= 0 and n <= 0:
            raise ValueError(
                f"Cannot reduce degrees further. Need at least {m + n + 2} "
                f"data points for a (0, 0) rational fit, but got {n_points}.")
        if reduce_denom and n > 0:
            n -= 1
        elif m > 0:
            m -= 1
        reduce_denom = not reduce_denom  # Alternate

    return (m, n)


class RationalFit:
    """Rational fit using Stabilized Sanathanan-Koerner iteration.
    
    The classical Sanathanan-Koerner iteration is a rational approximation
    method that multiplies the approximation by the denominator to linearize
    the problem. Then at each iteration, it introduces a weight to correct this
    linearization, leading to a weighted least squares problem.
    See https://ieeexplore.ieee.org/document/1105517.
    
    In the Stabilized Sanathanan-Koerner approximation, we use the Arnoldi
    polynomial basis, instead of Vandermonde mononomials, to represent the 
    numerator and denominator of the rational function to stabilize the
    fitting process, especially for higher degree rational functions.
    See https://arxiv.org/abs/2009.10803.
    """

    def __init__(self, x, y, degrees):
        """Initialize.
        
        parameters:
        -----------
        x: array-like, shape (n,)
            The input nodes.
        y: array-like, shape (n,)
            The target values at the nodes.
        degrees: (int, int)
            The degrees of the numerator and denominator of the rational function.
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if not isinstance(degrees, (tuple, list)):
            raise ValueError(
                "Degrees must be a tuple of (numerator_degree, denominator_degree).")
        if len(degrees) != 2:
            raise ValueError(
                "Degrees must be a tuple of (numerator_degree, denominator_degree).")
        if degrees[0] < 0 or degrees[1] < 0:
            raise ValueError("Degrees must be non-negative.")
        # numbr of data points should be greater than the number of parameters
        # to fit
        if is_underdetermined(degrees, len(x)):
            warnings.warn(
                "Number of data points must be greater than the number of "
                "parameters to fit (numerator degree + denominator degree + 1). "
                f"Got {len(x)} data points and degrees {degrees}.")
        self.x = x
        self.y = y
        self.degrees = degrees
        self.x_min, self.x_max = np.min(x), np.max(x)  # For scaling x
        self.a = None  # Coefficients for numerator basis functions, updated in fit() method
        self.b = None  # Coefficients for denominator basis functions, updated in fit() method

    def fit(self, max_iterations=100, tol=1e-7, verbose=False):
        """Fit using Stabilized Sanathanan-Koerner iteration.

        Solves the classical Sanathanan-Koerner iteration with Arnoldi basis 
        for better stability. 
        
        At each iteration k, we
        - generate the Arnoldi polynomial basis functions using w[k]=1/q[k-1] as
          the starting vector, see _arnoldi_basis. Note that we use the
          denominator from the previous iteration as the weight because we want
          to solve the weighted least square problem.
        - Our system to solve is y * Q * b = P * a, where P and Q are the 
          Arnoldi basis matrices for the numerator and denominator,
          respectively.
        - We can project out the numerator subspace from y * Q to get
          a least squares problem in b. The system to solve becomes
          (y * Q - P * (P^T * (y * Q))) * b = 0 using a = P^T * (y * Q).
        - We then solve for b using least squares, and reconstruct a using 
          a = P^T * (y * Q * b).        
        - we then compute p = P * a and q = Q * b to get the rational
          approximation r = p / q, and check for convergence.

        parameters:
        -----------
        max_iterations: int, default=100
            Maximum number of iterations for the fitting process.
        tol: float, default=1e-7
            Tolerance for convergence based on residual error between the
            rational approximation and the target values.
        verbose: bool, default=False
            If True, print iteration details.
        """
        # Scale x to [-1, 1] for better numerical stability
        x_scaled = _scale_x(self.x, self.x_min, self.x_max)
        m, n = self.degrees

        # Initialize denominator q^(0) = 1 (constant function)
        q = np.ones_like(self.x)

        # To keep track of fitting error
        fit_old = np.zeros_like(self.y)

        # History containers
        history = []

        for iteration in range(max_iterations):
            # weight w^(k) = 1 / q^(k-1)
            w = 1.0 / q

            # Generate Arnoldi basis with current weight w
            V, H, v0_norm = _arnoldi_basis(x_scaled, max(m, n), w)
            P, Q = V[:, :m+1], V[:, :n+1]

            # Generate yQ by multiplying y with the denominator basis 
            # matrix Q, so that the system to solve becomes
            # yQ * b = P * a
            yQ = self.y[:, np.newaxis] * Q
            # Now project out the numerator subspace from yQ
            # using P^T * yQ * b = a, and then find by solving
            # (yQ - P * (P^T *yQ)) b = 0
            yQ_projected = yQ -  P @ (P.T @ yQ)
            
            # Separate the first column (constant/base) from the rest
            target = -yQ_projected[:, 0]
            others = yQ_projected[:, 1:]

            # Solve for the remaining coefficients
            b_rest, _, _, _ = np.linalg.lstsq(others, target, rcond=1e-12)

            # Reconstruct b with the first coefficient forced to 1.0
            b = np.concatenate(([1.0], b_rest))
            a = P.T @ (self.y * (Q @ b))

            # Update denominator
            q = Q @ b

            # compute error
            p = P @ a  # Numerator
            r = p / q  # Rational approximation
            res_error = np.linalg.norm(r - self.y) / np.linalg.norm(self.y)
            fit_error = np.linalg.norm(r - fit_old)

            # save history
            hist_dict = {
                "iteration": iteration,
                "a": a.copy(),
                "b": b.copy(),
                "v0_norm": v0_norm,
                "H": H.copy(),
                "res_error": res_error,
                "fit_error": fit_error}
            history.append(hist_dict)

            if verbose:
                print(f"Iteration {iteration}: fit error = {fit_error:.2e}")

            # Check for convergence
            if fit_error < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} with fit error "
                          f"{fit_error:.2e}.")
                break
            # Update fit_old for the next iteration
            fit_old = r

        hist_best = min(history, key=lambda h: h["res_error"])
        self.a, self.b = hist_best["a"], hist_best["b"]
        self.H = hist_best["H"]
        # Store the norm of the initial weight vector w for later use in 
        # evaluating the basis
        self.v0_norm = hist_best["v0_norm"]
        self.history = history
    
    def predict(self, x_new):
        """Predict rational function values at new points x_new.
        
        parameters:
        -----------
        x_new: 1d array-like
            The new input points to predict the rational function values at.

        returns:
        --------
        r: 1d array
            The predicted rational function values at x_new.
        """
        x_new = np.asarray(x_new)
        is_scalar = x_new.ndim == 0
        x_new = np.atleast_1d(x_new)

        # Check input range
        if np.any(x_new < self.x_min) or np.any(x_new > self.x_max):
            warnings.warn("x_new contains points outside the training range.")

        # scale x_new using the same scaling as the original x
        x_new_scaled = _scale_x(x_new, self.x_min, self.x_max)

        # Evaluate the Arnoldi basis functions at x_new using the stored H
        # and v0_norm
        V = _eval_arnoldi_basis(
            x=x_new_scaled, H=self.H, degree=max(self.degrees),
            v0_norm=self.v0_norm)
        P = V[:, :self.degrees[0] + 1]
        Q = V[:, :self.degrees[1] + 1]
        
        # Compute the numerator and denominator at x_new using the
        # fitted coefficients
        p = P @ self.a
        q = Q @ self.b

        # Return the rational function values at x_new
        r = p / q

        return r.item() if is_scalar else r
    
    def __call__(self, x_new):
        """Predict rational function values at new points x_new.
        
        parameters:
        -----------
        x_new: 1d array-like
            The new input points to predict the rational function values at.

        returns:
        --------
        r: 1d array
            The predicted rational function values at x_new.
        """
        if self.a is None or self.b is None:
            raise ValueError(
                "Model is not fitted yet. Call fit() first.")
        return self.predict(x_new)
