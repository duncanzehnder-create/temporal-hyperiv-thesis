"""Train the baseline and G5 for one evaluation year."""
import argparse

import torch

from temporal_hyperiv import models
from temporal_hyperiv.data import Market

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', choices=['baseline', 'g5'])
p.add_argument('--year', type=int, choices=models.BASELINE_EPOCHS, required=True)
p.add_argument('--seed', type=int, choices=models.SEEDS, required=True)
p.add_argument('--data', default='data')
a = p.parse_args()
train = models.train_baseline if a.model == 'baseline' else models.train_g5
train(Market(a.data), a.year, a.seed, 'cuda' if torch.cuda.is_available() else 'cpu')
