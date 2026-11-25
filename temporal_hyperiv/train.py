"""HyperIV's loss and the training loop shared by the baseline and G5."""
import json
from pathlib import Path

import numpy as np
import torch

from hyperiv.trainer_util import aux_loss
from temporal_hyperiv.evaluate import daily_mae, predict

BATCH, TARGETS, CLIP, LEARNING_RATE, FINAL_RATE = 128, 1024, 0.1, 1e-4, 1e-5


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss(model, row, refs, x, iv):
    fit = (model(row, refs, x) - iv).square().mean()
    return fit + sum(aux_loss(lambda z, q: model(row, z, q).unsqueeze(-1), refs))


def fit(model, data, epochs, schedule, seed, out, validation=None):
    """Adam on a cosine schedule. Keeps the last weights, and the best ones if there is a validation year."""
    out = Path(out)
    out.mkdir(parents=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, schedule, eta_min=FINAL_RATE)
    best = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 999])).permutation(len(data))
        total, steps = 0.0, len(order) // BATCH
        for step in range(steps):
            idx = order[step * BATCH:(step + 1) * BATCH]
            optimiser.zero_grad(set_to_none=True)
            value = loss(model, *data.sample(idx, seed, epoch, TARGETS))
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, CLIP, error_if_nonfinite=True)
            optimiser.step()
            total += value.detach()
        cosine.step()
        record = {'epoch': epoch, 'loss': float(total) / steps}
        if validation is not None:
            record['iv_mae'] = float(np.mean(daily_mae(predict(model, validation), validation)))
            if record['iv_mae'] < best:
                best = record['iv_mae']
                torch.save(model.state_dict(), out / 'best.pt')
        with open(out / 'history.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
    torch.save(model.state_dict(), out / 'last.pt')
