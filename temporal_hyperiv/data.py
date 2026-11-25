"""Prepared surfaces, the walk-forward split and GPU batches."""
import json
from pathlib import Path

import numpy as np
import torch

HISTORY, MAX_GAP = 20, 7 * 86400


class Market:
    """All trading days. A day is identified by its row."""

    def __init__(self, folder):
        self.quotes = np.load(Path(folder) / 'quotes.npy', mmap_mode='r')
        days = json.loads((Path(folder) / 'surfaces.json').read_text())
        self.dates = [d['date'] for d in days]
        self.unix = np.array([d['unix'] for d in days])
        self.refs = np.array([d['refs'] for d in days])
        self.bounds = [(d['start'], d['stop']) for d in days]

    def panel(self, row):
        """The withheld quotes of one day (columns in prepare.COLUMNS)."""
        return self.quotes[slice(*self.bounds[row])]

    def rows(self, start='', end='9999'):
        """Days in [start, end) whose last 20 snapshots have no gap over seven days."""
        gaps = np.diff(self.unix)
        return [i for i in range(HISTORY - 1, len(self.dates))
                if start <= self.dates[i] < end and (gaps[i - HISTORY + 1:i] <= MAX_GAP).all()]

    def split(self, year):
        """Rows of the baseline fit, the G5 base fit, the G5 fit and the evaluation year."""
        y, g5 = f'{year}-01-01', f'{year - 4}-01-01'
        return {'train': self.rows(end=y), 'base': self.rows(end=g5), 'g5': self.rows(g5, y),
                'evaluate': self.rows(y, f'{year + 1}-01-01')}


class Batches:
    """Quotes of the chosen days on the device."""

    def __init__(self, market, rows, device):
        self.market, self.rows, self.device = market, np.array(rows), device
        panels = [np.array(market.panel(r)[:, :3], np.float32) for r in rows]
        self.counts = np.array([len(p) for p in panels])
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.values = torch.from_numpy(np.concatenate(panels)).to(device)
        self.refs = torch.tensor(market.refs[self.rows], dtype=torch.float32, device=device)
        self.row = torch.as_tensor(self.rows, device=device)

    def __len__(self):
        return len(self.rows)

    def sample(self, idx, seed, epoch, n):
        """n target quotes per day, drawn with replacement."""
        positions = [self.starts[i] + np.random.default_rng(np.random.SeedSequence([seed, epoch, int(self.rows[i])]))
                     .integers(self.counts[i], size=n) for i in idx]
        v = self.values[torch.as_tensor(np.array(positions), device=self.device)]
        i = torch.as_tensor(idx, device=self.device)
        return self.row[i], self.refs[i], v[..., :2], v[..., 2]

    def everything(self, idx):
        """Every target quote of the days, padded to the longest."""
        offsets = np.arange(self.counts[idx].max())
        positions = self.starts[idx, None] + np.minimum(offsets[None], self.counts[idx, None] - 1)
        v = self.values[torch.as_tensor(positions, device=self.device)]
        i = torch.as_tensor(idx, device=self.device)
        return self.row[i], self.refs[i], v[..., :2]
