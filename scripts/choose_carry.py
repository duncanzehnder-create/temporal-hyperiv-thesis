"""Choose the carry's damping and penalty on training years."""
import argparse

import numpy as np
import torch

from temporal_hyperiv import models
from temporal_hyperiv.carry import GRID_K, GRID_T, Carry, arbitrage_terms, carry_table, plain_step
from temporal_hyperiv.data import Batches, Market
from temporal_hyperiv.evaluate import TOLERANCE, daily_mae, predict

DAMPINGS, PENALTIES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1), (0.0, 0.1, 1.0, 10.0, 100.0)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', default='data')
a = p.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
market = Market(a.data)
grid = torch.cartesian_prod(*[torch.linspace(*g, device=device) for g in (GRID_K, GRID_T)])


def violations(model, data):
    """Mean of the butterfly and calendar rates on the penalty grid."""
    point = torch.vmap(lambda th, k, t: arbitrage_terms(model, th, k, t), in_dims=(None, 0, 0))
    thetas = [model.weights(r[None], z[None])[0].detach() for r, z in zip(data.row, data.refs)]
    return np.mean([float((point(th, grid[:, 0], grid[:, 1]) < -TOLERANCE).float().mean()) for th in thetas])


for year in (2020, 2021, 2022):
    rows = market.split(year)['g5']
    data = Batches(market, rows, device)
    for seed in models.SEEDS:
        base = models.load_g5_base(year, seed, device)
        mae = lambda model: daily_mae(predict(model, data), data).mean()
        errors = {d: mae(Carry(base, carry_table(base, market, rows, device, plain_step, damping=d))) for d in DAMPINGS}
        damping = min(errors, key=errors.get)
        rates = {e: violations(Carry(base, carry_table(base, market, rows, device, damping=damping, penalty=e)), data) for e in PENALTIES}
        limit = violations(base, data)
        fine = [e for e in PENALTIES if rates[e] <= limit]
        print(year, seed, 'no correction %.5f' % mae(base), errors, 'damping', damping, 'penalty', fine[0] if fine else min(rates, key=rates.get), flush=True)
