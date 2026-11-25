"""Yesterday's quotes: today's volatility read from yesterday's smile of the same expiry."""
import datetime as dt

import numpy as np


def _expiry(date, tau):
    return dt.date.fromisoformat(date).toordinal() + np.rint(np.asarray(tau) * 365).astype(int)


def yesterday(market, row):
    """Linear in log-moneyness within an expiry; linear in total variance between expiries."""
    today, p = dt.date.fromisoformat(market.dates[row]).toordinal(), market.panel(row)
    before = np.vstack([market.panel(row - 1)[:, :3], market.refs[row - 1]])
    expiry, wanted = _expiry(market.dates[row - 1], before[:, 1]), _expiry(market.dates[row], p[:, 1])
    smiles = {}
    for e in np.unique(expiry):
        q = before[expiry == e]
        q = q[np.argsort(q[:, 0])]
        smiles[e] = q[:, 0], q[:, 2]
    live = np.array([e for e in smiles if e > today])
    out = np.empty(len(p))
    for e in np.unique(wanted):
        here, k = wanted == e, p[wanted == e, 0]
        below, above = live[live < e], live[live > e]
        if e in smiles:
            out[here] = np.interp(k, *smiles[e])
        elif len(below) and len(above):
            a, b = below[-1], above[0]
            t, ta, tb = ((x - today) / 365 for x in (e, a, b))
            wa, wb = ta * np.interp(k, *smiles[a]) ** 2, tb * np.interp(k, *smiles[b]) ** 2
            out[here] = np.sqrt(np.maximum(wa + (wb - wa) * (t - ta) / (tb - ta), 1e-12) / t)
        else:
            out[here] = np.interp(k, *smiles[above[0] if len(above) else below[-1]])
    return np.maximum(out, 1e-4)
