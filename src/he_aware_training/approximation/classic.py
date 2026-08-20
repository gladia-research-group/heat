import functools
import math
import numpy as np
from scipy.optimize import differential_evolution, root


def polyval_torch(x, coeffs):
    val = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        val = val * x + coeffs[i]
    return val

def chebyshev_coeffs(func, a, b, deg):
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    return np.polynomial.chebyshev.chebinterpolate(lambda t: func(half * t + mid), deg)


def chebyshev_series_torch(x, coeffs, a, b):
    """Evaluate sum_i coeffs[i]*T_i(y) with y=(2x-(a+b))/(b-a) via Clenshaw.
    """
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    y = (x - mid) / half
    n = coeffs.shape[0] - 1
    if n <= 0:
        return y * 0 + coeffs[0]
    if n == 1:
        return coeffs[0] + coeffs[1] * y
    two_y = 2.0 * y
    b1 = y * 0 + coeffs[n]   # zeros broadcast over x, plus the leading coeff
    b2 = y * 0
    for k in range(n - 1, 0, -1):
        b0 = coeffs[k] + two_y * b1 - b2
        b2, b1 = b1, b0
    return coeffs[0] + y * b1 - b2


def get_chebyshev_nodes(a, b, n):
    # starting from 1
    k = np.arange(1, n + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

@functools.lru_cache(maxsize=None)
def find_best_rational_coeffs(target_func, a, b, n, m, nodes_selection="chebyshev"):
    if nodes_selection == "chebyshev":
        x_train = get_chebyshev_nodes(a, b, 200)
    elif nodes_selection == "uniform":
        x_train = np.linspace(a, b, 200)
    elif nodes_selection == "log":
        x_train = np.geomspace(a, b, 200)
    else:
        raise ValueError(f"Unknown nodes_selection: {nodes_selection}")

    y_train = target_func(x_train)

    def objective(params):
        p = params[: n + 1]
        q_high = params[n + 1 :]
        q = np.concatenate(([1.0], q_high))

        num = np.polynomial.polynomial.polyval(x_train, p)
        den = np.polynomial.polynomial.polyval(x_train, q)

        if np.min(den) <= 1e-4:
            return 1e10

        approx = num / den
        rel_error = np.abs((approx - y_train) / y_train)
        return np.max(rel_error)

    bounds = [(-100, 100)] * (n + 1)
    bounds += [(0.0, 1.0)] * m

    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=2000,
        popsize=60,
        tol=1e-9,
    )

    p_final = result.x[: n + 1]
    q_final = np.concatenate(([1.0], result.x[n + 1 :]))

    q_min = np.min(np.polynomial.polynomial.polyval(np.linspace(a, b, 1000), q_final))
    q_max = np.max(np.polynomial.polynomial.polyval(np.linspace(a, b, 1000), q_final))

    return p_final, q_final, q_min, q_max


@functools.lru_cache(maxsize=None)
def rational_remez(target_func, a, b, n, m, max_iter=50, tol=1e-12, grid_size=5000):
    num_points = n + m + 2

    x_nodes = np.sort(get_chebyshev_nodes(a, b, num_points))

    cheb_dense = np.cos((2*np.arange(1, grid_size+1) - 1) / (2*grid_size) * np.pi)
    x_dense = np.sort(0.5 * (b - a) * cheb_dense + 0.5 * (b + a))

    p_coeffs = np.zeros(n + 1)
    q_coeffs = np.zeros(m + 1)
    q_coeffs[0] = 1.0

    print(f"Starting Remez (n={n}, m={m})...")

    for iteration in range(max_iter):
        y_nodes = target_func(x_nodes)

        def equations(vars):
            p = vars[: n + 1]
            q_high = vars[n + 1 : n + m + 1]
            E = vars[-1]

            q = np.concatenate(([1.0], q_high))

            P_val = np.polynomial.polynomial.polyval(x_nodes, p)
            Q_val = np.polynomial.polynomial.polyval(x_nodes, q)

            signs = (-1.0) ** np.arange(num_points)

            return P_val - y_nodes * Q_val * (1.0 + signs * E)

        guess = np.zeros(num_points)
        if iteration == 0:
            guess[0] = np.mean(y_nodes)
        else:
            guess[: n + 1] = p_coeffs
            guess[n + 1 : n + m + 1] = q_coeffs[1:]

        sol = root(equations, guess, method="lm")

        if not sol.success and iteration > 0:
            print("  Convergence stalled in solver. Returning previous best.")
            break

        p_coeffs = sol.x[: n + 1]
        q_coeffs = np.concatenate(([1.0], sol.x[n + 1 : n + m + 1]))
        E_curr = sol.x[-1]

        y_dense = target_func(x_dense)
        num = np.polynomial.polynomial.polyval(x_dense, p_coeffs)
        den = np.polynomial.polynomial.polyval(x_dense, q_coeffs)

        if np.min(np.abs(den)) < 1e-9:
            print("  Warning: Denominator near zero. Pole detected.")

        approx = num / den
        rel_error = (approx - y_dense) / y_dense
        abs_rel_error = np.abs(rel_error)

        max_err_idx = np.argmax(abs_rel_error)
        max_err_val = abs_rel_error[max_err_idx]
        x_extremum = x_dense[max_err_idx]
        ext_sign = np.sign(rel_error[max_err_idx])

        diff = abs(max_err_val - abs(E_curr))

        if diff < tol:
            break

        n_num = np.polynomial.polynomial.polyval(x_nodes, p_coeffs)
        n_den = np.polynomial.polynomial.polyval(x_nodes, q_coeffs)
        node_errs = (n_num / n_den - y_nodes) / y_nodes
        node_signs = np.sign(node_errs)

        idx = np.searchsorted(x_nodes, x_extremum)

        if idx == 0:
            if np.sign(ext_sign) == np.sign(node_signs[0]):
                x_nodes[0] = x_extremum
            else:
                x_nodes = np.insert(x_nodes, 0, x_extremum)[:-1]
        elif idx == num_points:
            if np.sign(ext_sign) == np.sign(node_signs[-1]):
                x_nodes[-1] = x_extremum
            else:
                x_nodes = np.append(x_nodes, x_extremum)[1:]
        else:
            s_left = node_signs[idx - 1]
            s_right = node_signs[idx]

            if np.sign(ext_sign) == np.sign(s_left):
                x_nodes[idx - 1] = x_extremum
            elif np.sign(ext_sign) == np.sign(s_right):
                x_nodes[idx] = x_extremum

        x_nodes = np.sort(x_nodes)

    x_dense = np.linspace(a, b, 1000)

    final_den_vals = np.polynomial.polynomial.polyval(x_dense, q_coeffs)
    q_min = np.min(final_den_vals)
    q_max = np.max(final_den_vals)

    return p_coeffs, q_coeffs, q_min, q_max

def compute_optimal_linear_init(d_min, d_max):
    """
    From the paper
    """
    alpha = 4.0 / (d_min + d_max)
    beta = alpha**2 / 4.0
    return alpha, beta


def compute_optimal_chebyshev_init(d_min, d_max):
    s = d_min + d_max
    beta = 8.0 / (s * s + 4.0 * d_min * d_max)
    alpha = beta * s
    return alpha, beta

def compute_optimal_linear_init_minimax(d_min, d_max):
    alpha = (d_min ** .5 + d_max ** .5) ** 2 / (2 * d_min * d_max)
    beta = 1 / (d_min * d_max)
    return alpha, beta

def taylor_inv_sqrt_coeffs(z0):
    """
    Degree-3 Taylor coefficients of 1/sqrt(z) centered at z0, shifted basis.
    """
    z0_sqrt = np.sqrt(z0)
    a0 =  1.0 / z0_sqrt
    a1 = -1.0 / (2.0 * z0 * z0_sqrt)
    a2 =  3.0 / (8.0 * z0**2 * z0_sqrt)
    a3 = -5.0 / (16.0 * z0**3 * z0_sqrt)
    return np.array([a0, a1, a2, a3])


@functools.lru_cache(maxsize=None)
def taylor_inv_sqrt(z0=0.5):
    coeffs = taylor_inv_sqrt_coeffs(z0)
    return coeffs, z0


def mean_error_to_bits(mean_error, a, b):
    return np.ceil(np.log2(np.abs(b-a)/mean_error + 1))


if __name__ == "__main__":
    def goldschmidt(num, den, alpha, beta, iters=3):
        def goldschmidt_step(N, D, F):
            N_next = N * F
            D_neg = D * F
            F_next = 2.0 + D_neg
            return N_next, D_neg, F_next

        F = alpha - beta * den
        N = num * F
        D = den * -F
        F = 2.0 + D
        for i in range(iters - 1):
            N, D, F = goldschmidt_step(N, D, F)
        
        return N

    def newton(X, y, iters=3):
        def newton_step(X, y):
            return .5 * y * (3.0 - X * y * y)
        
        for i in range(iters):
            y = newton_step(X, y)
        return y
    
    def newton_div(X, y, iters=3):
        def newton_step(X, y):
            return y * (2.0 - X * y)
        
        for i in range(iters):
            y = newton_step(X, y)
        return y


    # R_MIN = -64.0
    # R_MAX = 0.0
    
    # N_DEG = 4
    # M_DEG = 4
    # K_SQUARING = 4
    # T_SQUARING = 0
    # ITERS = 4
    # p_opt, q_opt, q_min, q_max = rational_remez(lambda x: np.exp(x), R_MIN / (2**(K_SQUARING+T_SQUARING)), R_MAX / (2**(K_SQUARING+T_SQUARING)), N_DEG, M_DEG)

    # X = np.linspace(R_MIN, R_MAX, 1000).astype(np.float64)

    # alpha, beta = compute_optimal_linear_init_minimax(q_min ** (2**T_SQUARING), q_max ** (2**T_SQUARING))

    # num = np.polynomial.polynomial.polyval(X / (2**(K_SQUARING+T_SQUARING)), p_opt) ** (2**T_SQUARING)
    # den = np.polynomial.polynomial.polyval(X / (2**(K_SQUARING+T_SQUARING)), q_opt) ** (2**T_SQUARING)
        
    # Y_approx = goldschmidt(num, den, alpha, beta, iters=ITERS) ** (2**K_SQUARING)
    # Y_true = np.exp(X)
    
    # errors = np.abs(Y_true - Y_approx)
    # max_err = np.max(errors)
    # mean_err = np.mean(errors)
    # n_bits = mean_error_to_bits(mean_err, np.exp(R_MIN), np.exp(R_MAX))

    
    # print(f"Maximum Absolute Error for e^x on [{R_MIN}, {R_MAX}]: {max_err}")
    # print(f"Mean Absolute Error for e^x on [{R_MIN}, {R_MAX}]: {mean_err}")
    # print(f"Estimated bits of precision for e^x on [{R_MIN}, {R_MAX}]: {n_bits}")

    R_MIN = 1.0
    R_MAX = 126 ** 2 * 9.8491 + 1
    
    N_DEG = 3
    M_DEG = 1
    func = lambda x: x ** -.5
    p_opt, q_opt, q_min, q_max = rational_remez(func, R_MIN, R_MAX, N_DEG, M_DEG)

    print(p_opt)
    print(q_opt)

    X = np.linspace(R_MIN, R_MAX, 1000).astype(np.float64)

    alpha, beta = compute_optimal_linear_init(q_min, q_max)

    num = np.polynomial.polynomial.polyval(X, p_opt)
    den = np.polynomial.polynomial.polyval(X, q_opt)
        
    Y_approx = goldschmidt(num, den, alpha, beta, iters=14)
    Y_approx = newton(X, Y_approx, iters=4)
    Y_true = func(X)
    
    errors = np.abs(Y_true - Y_approx)
    max_err = np.max(errors)
    mean_err = np.mean(errors)
    n_bits = mean_error_to_bits(mean_err, func(R_MIN), func(R_MAX))

    print(f"Maximum Absolute Error for x^-0.5 on [{R_MIN}, {R_MAX}]: {max_err}")
    print(f"Mean Absolute Error for x^-0.5 on [{R_MIN}, {R_MAX}]: {mean_err}")
    print(f"Estimated bits of precision for x^-0.5 on [{R_MIN}, {R_MAX}]: {n_bits}")


    # R_MIN = 0.04
    # R_MAX = 1.0
    
    # N_DEG = 2
    # M_DEG = 0
    # p_opt, _, _, _ = rational_remez(lambda x: 1 / x, R_MIN, R_MAX, N_DEG, M_DEG)

    # X = np.linspace(R_MIN, R_MAX, 1000).astype(np.float64)

    # num = np.polynomial.polynomial.polyval(X, p_opt)
        
    # Y_approx = newton_div(X, num, iters=4)
    # Y_true = 1 / X
    
    # errors = np.abs(Y_true - Y_approx)
    # max_err = np.max(errors)
    # mean_err = np.mean(errors)
    # n_bits = mean_error_to_bits(mean_err, 1 / R_MIN, 1 / R_MAX)

    # print(f"Maximum Absolute Error for 1/x on [{R_MIN}, {R_MAX}]: {max_err}")
    # print(f"Mean Absolute Error for 1/x on [{R_MIN}, {R_MAX}]: {mean_err}")
    # print(f"Estimated bits of precision for 1/x on [{R_MIN}, {R_MAX}]: {n_bits}")