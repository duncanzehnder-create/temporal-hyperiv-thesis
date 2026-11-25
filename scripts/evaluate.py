"""Score every model on one evaluation year."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporal_hyperiv import models
from temporal_hyperiv.benchmark import yesterday
from temporal_hyperiv.data import Batches, Market
from temporal_hyperiv.evaluate import arbitrage, autocorrelation, binned_mae, daily_mae, predict, stress_days

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--year', type=int, choices=models.BASELINE_EPOCHS, required=True)
p.add_argument('--data', default='data')
a = p.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
market = Market(a.data)
data = Batches(market, market.split(a.year)['evaluate'], device)
stress = np.isin(data.rows, list(stress_days(market, data.rows)))


def summary(predictions, model=None):
    daily = daily_mae(predictions, data)
    return {'iv_mae': daily.mean(), 'stress': daily[stress].mean(), 'calm': daily[~stress].mean(), 'daily': daily,
            'bins': binned_mae(predictions, data), 'arbitrage': arbitrage(model, data) if model else {}}


def average(items):
    first = items[0]
    return {k: average([i[k] for i in items]) for k in first} if isinstance(first, dict) else np.mean(items, axis=0).tolist()


runs = {'baseline': [], 'carry': [], 'g5': []}
correlation = []
for seed in models.SEEDS:
    built = {'baseline': models.load_baseline(a.year, seed, device), 'carry': models.load_carry(market, a.year, seed, device),
             'g5': models.load_g5(market, a.year, seed, device)}
    for name, model in built.items():
        predictions = predict(model, data)
        runs[name].append(summary(predictions, model))
        if name == 'baseline':
            correlation.append(autocorrelation(predictions, data))
out = {name: average(r) for name, r in runs.items()}
out['yesterday'] = average([summary([yesterday(market, r) for r in data.rows])])
out['autocorrelation'] = np.ma.masked_invalid(correlation).mean(axis=0).tolist()
Path('runs').mkdir(exist_ok=True)
Path(f'runs/scores-{a.year}.json').write_text(json.dumps(out, indent=1))
for name in ('baseline', 'carry', 'g5', 'yesterday'):
    print('%-10s %.5f' % (name, out[name]['iv_mae']))
