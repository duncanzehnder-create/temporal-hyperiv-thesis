"""Train and load the models of one evaluation year."""
from pathlib import Path

import torch

from temporal_hyperiv.baseline import HyperIV
from temporal_hyperiv.carry import Carry, carry_table
from temporal_hyperiv.data import Batches
from temporal_hyperiv.fields import error_fields
from temporal_hyperiv.gru import G5
from temporal_hyperiv.train import fit, set_seed

SEEDS = 17, 43, 89
BASELINE_EPOCHS = {2020: 500, 2021: 500, 2022: 500, 2023: 454}
G5_EPOCHS = {2020: 498, 2021: 49, 2022: 209, 2023: 407}
RUNS = Path('runs')


def folder(name, year, seed):
    return RUNS / f'{name}-{year}-seed-{seed}'


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.to(device).eval()


def train_hyperiv(market, rows, epochs, seed, out, device, validation=None):
    set_seed(seed)
    fit(HyperIV().to(device), Batches(market, rows, device), epochs, 500, seed, out,
        validation and Batches(market, validation, device))


def train_baseline(market, year, seed, device):
    s = market.split(year)
    train_hyperiv(market, s['train'], BASELINE_EPOCHS[year], seed, folder('baseline', year, seed), device,
                  s['evaluate'] if year < 2023 else None)


def train_g5(market, year, seed, device):
    s = market.split(year)
    train_hyperiv(market, s['base'], 500, seed, folder('g5-base', year, seed), device)
    model = new_g5(market, year, seed, device, s['g5'][-1])
    fit(model, Batches(market, s['g5'], device), G5_EPOCHS[year], 600, seed, folder('g5', year, seed))


def new_g5(market, year, seed, device, last_row):
    base = load_g5_base(year, seed, device)
    inputs = error_fields(base, market, last_row, device, market.split(year)['g5'])
    set_seed(seed)
    return G5(base, market, *inputs, device)


def load_baseline(year, seed, device):
    return load(HyperIV(), folder('baseline', year, seed) / ('best.pt' if year < 2023 else 'last.pt'), device)


def load_g5_base(year, seed, device):
    return load(HyperIV(), folder('g5-base', year, seed) / 'last.pt', device)


def load_carry(market, year, seed, device, **settings):
    base = load_baseline(year, seed, device)
    return Carry(base, carry_table(base, market, market.split(year)['evaluate'], device, **settings)).eval()


def load_g5(market, year, seed, device):
    return load(new_g5(market, year, seed, device, market.split(year)['evaluate'][-1]), folder('g5', year, seed) / 'last.pt', device)
