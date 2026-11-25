"""Scores on the withheld quotes and arbitrage checks on a grid."""
import math

import numpy as np
import torch
from scipy.special import ndtr

from hyperiv.trainer_util import calc_g_func, get_grad

K_BINS, DAY_BINS = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf], [0, 30, 90, 180, 365, 731]
GRID, TOLERANCE = {'k_start': -1.49, 'k_end': 0.49, 'k_num': 81, 't_start': 0.015, 't_end': 1.995, 't_num': 21}, 1e-6


def predict(model, data, batch=8):
    """Predicted volatility of every withheld quote, one array per day."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(data), batch):
            idx = np.arange(start, min(start + batch, len(data)))
            values = model(*data.everything(idx)).cpu().numpy()
            out += [values[j, :data.counts[i]].astype(float) for j, i in enumerate(idx)]
    return out


def daily_mae(predictions, data):
    return np.array([np.abs(p - data.market.panel(r)[:, 2]).mean() for p, r in zip(predictions, data.rows)])


def binned_mae(predictions, data):
    """Daily-mean MAE within moneyness and maturity bins."""
    out = {}
    for name, column, scale, edges in (('moneyness', 0, 1, K_BINS), ('days', 1, 365, DAY_BINS)):
        for low, high in zip(edges, edges[1:]):
            days = []
            for p, r in zip(predictions, data.rows):
                q = data.market.panel(r)
                inside = (q[:, column] * scale >= low) & (q[:, column] * scale < high)
                if inside.any():
                    days.append(np.abs(p - q[:, 2])[inside].mean())
            out[f'{name} [{low}, {high})'] = float(np.mean(days))
    return out


def stress_days(market, rows, fraction=0.1):
    """The days with the largest move in the 30-day at-the-money reference volatility."""
    moves = [abs(market.refs[r][4][2] - market.refs[r - 1][4][2]) for r in rows]
    return set(np.array(rows)[np.argsort(moves)[::-1][:math.ceil(fraction * len(rows))]])


def arbitrage(model, data, batch=8):
    """Share of grid points failing each check, averaged over days."""
    model.eval()
    shape, rates = (-1, GRID['k_num'], GRID['t_num']), []
    for start in range(0, len(data), batch):
        row, refs = data.row[start:start + batch], data.refs[start:start + batch]
        kt, s, s_k, s_t, s_kk = get_grad(lambda z, q: model(row, z, q).unsqueeze(-1), refs, **GRID)
        k, t = kt[..., 0], kt[..., 1]
        butterfly, calendar = calc_g_func(k, t, s, s_k, s_kk), s + 2 * t * s_t
        s, k, t = (v.detach().double().cpu().numpy().reshape(shape) for v in (s, k, t))
        strike, sd = np.exp(k), s * np.sqrt(t)
        d1 = (-k + 0.5 * sd ** 2) / sd
        slope = np.diff(ndtr(d1) - strike * ndtr(d1 - sd), axis=1) / np.diff(strike, axis=1)
        for j in range(len(refs)):
            rates.append([float((butterfly[j] < -TOLERANCE).float().mean()), float((np.diff(slope[j], axis=0) < -TOLERANCE).mean()),
                          float((calendar[j] < -TOLERANCE).float().mean()), float((slope[j] > TOLERANCE).mean())])
    return dict(zip(('butterfly', 'convexity', 'calendar', 'monotonicity'), np.mean(rates, axis=0).tolist()))


def autocorrelation(predictions, data):
    """Lag-one autocorrelation of the daily mean error in each moneyness (rows) and maturity (columns) bin."""
    means = np.full((len(data), 5, 5), np.nan)
    for i, (p, r) in enumerate(zip(predictions, data.rows)):
        q = data.market.panel(r)
        a = np.searchsorted(K_BINS, q[:, 0], side='right') - 1
        b = np.searchsorted(DAY_BINS, q[:, 1] * 365, side='right') - 1
        for j, l in {*zip(a, b)}:
            means[i, j, l] = (p - q[:, 2])[(a == j) & (b == l)].mean()
    out = np.full((5, 5), np.nan)
    for j, l in np.ndindex(5, 5):
        x, y = means[:-1, j, l], means[1:, j, l]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 10:
            out[j, l] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out
