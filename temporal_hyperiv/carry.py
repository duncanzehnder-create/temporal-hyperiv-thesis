"""Weight-space carry: yesterday's error as a change to the 337 weights."""
import numpy as np
import torch

from temporal_hyperiv.baseline import HyperIV
from temporal_hyperiv.fields import cell_means, day_errors

DAMPING, PENALTY, ITERATIONS = 0.01, 1.0, 6
GRID_K, GRID_T = (-1.5, 0.5, 71), (0.01, 2.0, 17)


def _system(model, theta, k, tau, error, count):
    """Weighted Jacobian and error on the occupied cells."""
    used = count > 0
    points = torch.stack([k[used], tau[used]], -1)
    J = torch.func.jacrev(lambda th: model.decode(th[None], points[None])[0])(theta)
    w = (count[used] / count[used].mean()).sqrt()
    return w[:, None] * J, w * error[used]


def plain_step(model, theta, k, tau, error, count, damping=DAMPING):
    J, e = _system(model, theta, k, tau, error, count)
    K = J @ J.T
    return -J.T @ torch.linalg.solve(K + damping * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype, device=K.device), e)


def arbitrage_terms(model, theta, k, tau):
    """Butterfly function and calendar term at one point."""
    sigma = lambda k, tau: model.decode(theta[None], torch.stack([k, tau])[None, None])[0, 0]
    s = sigma(k, tau)
    s_k, s_t = torch.func.grad(sigma, argnums=(0, 1))(k, tau)
    s_kk = torch.func.grad(torch.func.grad(sigma))(k, tau)
    return torch.stack([(1 - k * s_k / s) ** 2 - (0.5 * tau * s * s_k) ** 2 + tau * s * s_kk, s + 2 * tau * s_t])


def arbitrage_step(model, theta, k, tau, error, count, damping=DAMPING, penalty=PENALTY):
    """The step with a penalty on linearised butterfly and calendar violations (active set)."""
    J, e = _system(model, theta, k, tau, error, count)
    H = J.T @ J + damping * J.square().sum(1).mean() * torch.eye(len(theta), dtype=J.dtype, device=J.device)
    b = -J.T @ e
    grid = torch.cartesian_prod(*[torch.linspace(*g, dtype=theta.dtype, device=theta.device) for g in (GRID_K, GRID_T)])
    terms = lambda th, k, tau: arbitrage_terms(model, th, k, tau)
    g = torch.vmap(terms, in_dims=(None, 0, 0))(theta, grid[:, 0], grid[:, 1]).reshape(-1)
    G = torch.vmap(torch.func.jacrev(terms), in_dims=(None, 0, 0))(theta, grid[:, 0], grid[:, 1]).reshape(len(g), -1)
    c, u, active = penalty * len(e) / len(g), torch.linalg.solve(H, b), None
    for _ in range(ITERATIONS):
        violated = g + G @ u < 0
        if active is not None and torch.equal(violated, active):
            break
        active = violated
        A = G[active]
        u = torch.linalg.solve(H + c * A.T @ A, b - c * A.T @ g[active])
    return u


def carry_table(model, market, rows, device, step=arbitrage_step, **settings):
    """Weight change for each row, from the day before alone. Other rows stay zero."""
    double = lambda v: torch.as_tensor(v, dtype=torch.float64, device=device)
    table = torch.zeros(max(rows) + 1, 337, device=device)
    for row in rows:
        p = market.panel(row - 1)
        refs = torch.tensor(market.refs[row - 1][None], dtype=torch.float32, device=device)
        with torch.no_grad():
            theta = model.weights(None, refs)[0].double()
        count, k, tau, error = cell_means(p[:, 0], p[:, 1], p[:, 0], p[:, 1], day_errors(model, market, row - 1, device))
        table[row] = step(model, theta, double(k), double(tau), double(error), double(count), **settings).float()
    return table


class Carry(HyperIV):
    """Baseline weights plus the day's stored change."""

    def __init__(self, base, table):
        super().__init__()
        self.model, self.table = base.model, table

    def weights(self, row, refs):
        return super().weights(row, refs) + self.table[row]
