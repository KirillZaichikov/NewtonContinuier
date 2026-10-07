import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "Newton.py"
SPEC = importlib.util.spec_from_file_location("newton_module", MODULE_PATH)
newton = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(newton)


def test_newton_solver_linear_system():
    solver = newton.NewtonSolver(dim=2, tol=1e-12, max_iter=10, eps=1e-8)

    def G(x, alpha, beta):
        return np.array([2.0 * x[0] - 1.0, 3.0 * x[1] - 2.0])

    x0 = np.array([0.0, 0.0])
    sol, iters, success, error = solver.solve(G, x0, 0.0, 0.0)

    assert success is True
    assert iters <= solver.max_iter
    np.testing.assert_allclose(sol, np.array([0.5, 2.0 / 3.0]), atol=1e-9)
    assert np.isfinite(error)


def test_finite_difference_jacobian_approximation():
    solver = newton.NewtonSolver(dim=2, eps=1e-6)

    def G(x, alpha, beta):
        return np.array([
            x[0] ** 2 + x[1] - 1.0,
            x[0] + x[1] ** 2 - 1.0,
        ])

    x = np.array([1.0, 1.0])
    jac = solver.finite_difference_jacobian(G, x, 0.0, 0.0)
    expected = np.array([[2.0, 1.0], [1.0, 2.0]])

    np.testing.assert_allclose(jac, expected, rtol=1e-4, atol=1e-4)


def test_fixed_point_continuation_g():
    F = lambda x, alpha, beta: np.array([
        x[0] + alpha,
        x[1] + beta,
        x[2] + alpha + beta,
    ])
    continuation = newton.FixedPointContinuation(F, dim=3)
    x = np.array([1.0, 2.0, 3.0])
    g = continuation.G(x, 0.5, -0.25)

    np.testing.assert_allclose(g, np.array([0.5, -0.25, 0.25]))


def test_ad_to_mg_and_test_f_produce_valid_shapes():
    x = np.array([3.141592653589793, 0.869986293341021, -0.645116618619213])
    alpha = 1.570796326794897
    beta = 670.190050063955937

    mg = newton.ADtoMG(x, alpha, beta)
    assert mg.shape == (6,)
    assert np.all(np.isfinite(mg))

    fx = newton.test_F(x, alpha, beta)
    assert fx.shape == (3,)
    assert np.all(np.isfinite(fx))


def test_calc_r_vec_returns_vector():
    gamma = np.array([0.2, 0.5, 0.8660254])
    r = newton.calc_r_vec(1.0, 2.0, 1.0, gamma, 0.0)

    assert r.shape == (3,)
    assert np.all(np.isfinite(r))
