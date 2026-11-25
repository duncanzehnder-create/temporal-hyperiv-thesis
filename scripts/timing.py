"""Time to generate one surface's weights when the day's quotes arrive."""
import argparse
import random
import statistics
import time

import torch
from torch.nn.functional import linear

from temporal_hyperiv import models
from temporal_hyperiv.carry import Carry, carry_table
from temporal_hyperiv.data import Batches, Market

WARMUP, CALLS = 200, 1500

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', default='data')
a = p.parse_args()
device, year, seed = 'cuda', 2023, 17
market = Market(a.data)
row = market.split(2022)['evaluate'][-1]
day = Batches(market, [row], device)
base = models.load_baseline(year, seed, device)
carry = Carry(base, carry_table(base, market, [row], device)).eval()
g5 = models.load(models.new_g5(market, year, seed, device, row), models.folder('g5', year, seed) / 'last.pt', device)
with torch.no_grad():
    layer, bias = g5.last_layer(day.row), g5.model._hyper.fc2.bias
    steps = {'baseline': lambda: base.weights(day.row, day.refs), 'carry': lambda: carry.weights(day.row, day.refs),
             'g5': lambda: linear(g5.encode(day.refs), layer, bias)}
    times = {n: [] for n in steps}
    for i in range(WARMUP + CALLS):
        for n in random.sample(list(steps), len(steps)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            steps[n]()
            torch.cuda.synchronize()
            if i >= WARMUP:
                times[n].append(1e3 * (time.perf_counter() - start))
for n, t in times.items():
    print('%-10s %.3f ms' % (n, statistics.median(t)))
