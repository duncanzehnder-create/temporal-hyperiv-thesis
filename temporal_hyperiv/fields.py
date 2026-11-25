"""The previous days' errors on a fixed 10 x 8 grid."""
import numpy as np
import torch

K_EDGES = np.array([-0.40, -0.25, -0.15, -0.08, -0.03, 0.0, 0.03, 0.08, 0.15])
DAY_EDGES = np.array([10, 21, 45, 75, 120, 200, 365])
CELLS = (len(K_EDGES) + 1) * (len(DAY_EDGES) + 1)


def cells(k, tau):
    return np.searchsorted(K_EDGES, k, side='right') * (len(DAY_EDGES) + 1) + np.searchsorted(DAY_EDGES, np.asarray(tau) * 365, side='right')


def cell_means(k, tau, *values):
    """Quotes per cell, and the mean of each value per cell (zero where empty)."""
    c = cells(k, tau)
    n = np.bincount(c, minlength=CELLS).astype(float)
    return [n] + [np.bincount(c, weights=v, minlength=CELLS) / np.maximum(n, 1) for v in values]


@torch.no_grad()
def day_errors(model, market, row, device):
    """The baseline's error on every withheld quote of one day."""
    p = market.panel(row)
    refs = torch.tensor(market.refs[row][None], dtype=torch.float32, device=device)
    x = torch.tensor(p[None, :, :2], dtype=torch.float32, device=device)
    return model(None, refs, x)[0].double().cpu().numpy() - p[:, 2]


def error_fields(model, market, last_row, device, standardise_on):
    """G5's inputs for every day: error field, mask and the 162 forward-move features."""
    n = last_row + 1
    field, mask, moved, moved_mask, extra = (np.zeros((n, w)) for w in (CELLS, CELLS, CELLS, CELLS, 2))
    errors, log_forward = None, None
    for t in range(n):
        p = market.panel(t)
        now = np.log(p[np.argmin(np.abs(p[:, 1] - 30 / 365)), 4])
        if t > 0:
            y, gap, move = market.panel(t - 1), (market.unix[t] - market.unix[t - 1]) / 86400, now - log_forward
            tau = y[:, 1] - gap / 365
            live = tau > 0
            count, moved[t] = cell_means(y[live, 0] - move, tau[live], errors[live])
            moved_mask[t], extra[t] = count > 0, (move, gap)
        errors, log_forward = day_errors(model, market, t, device), now
        count, field[t] = cell_means(p[:, 0], p[:, 1], errors)
        mask[t] = count > 0
    features = np.hstack([moved, moved_mask, extra])
    mean, sd = features[standardise_on].mean(0), features[standardise_on].std(0)
    features = np.where(sd > 1e-6, np.clip((features - mean) / np.where(sd > 1e-6, sd, 1), -10, 10), 0.0)
    return field, mask, features
