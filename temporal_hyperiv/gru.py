"""G5: a GRU over five days of error fields that changes the last layer of a fixed HyperIV."""
import torch
from torch import nn

from temporal_hyperiv.baseline import HyperIV
from temporal_hyperiv.fields import CELLS

DAYS, WIDTH, HIDDEN, RANK, EMBED, THETA, FEATURES = 5, 64, 16, 4, 128, 337, 2 * CELLS + 2


class GRU(nn.Module):
    """torch.nn.GRU's equations, with log(1 + age in days) added to each input."""

    def __init__(self):
        super().__init__()
        self.ih, self.hh = nn.Linear(WIDTH + 1, 3 * WIDTH), nn.Linear(WIDTH, 3 * WIDTH)
        for p in self.parameters():
            nn.init.uniform_(p, -WIDTH ** -0.5, WIDTH ** -0.5)

    def forward(self, x, age):
        x = torch.cat([x, torch.log1p(age).unsqueeze(-1)], dim=-1)
        h = x.new_zeros(len(x), WIDTH)
        for t in range(x.shape[1]):
            (ir, iz, i_n), (hr, hz, hn) = self.ih(x[:, t]).chunk(3, -1), self.hh(h).chunk(3, -1)
            r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            h = (1 - z) * torch.tanh(i_n + r * hn) + z * h
        return h


class G5(HyperIV):
    def __init__(self, base, market, field, mask, features, device):
        super().__init__()
        self.model = base.model.requires_grad_(False)
        self.project, self.project_extra, self.encoder = nn.Linear(2 * CELLS, WIDTH), nn.Linear(FEATURES, WIDTH), GRU()
        self.generator = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, RANK * (EMBED + THETA)))
        last = self.generator[-1]
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            nn.init.normal_(last.weight[:RANK * EMBED], std=0.02)
            last.weight[RANK * EMBED:].zero_()
        single = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
        self.inputs, self.features = torch.cat([single(field), single(mask)], dim=-1), single(features)
        self.unix = torch.as_tensor(market.unix, device=device)
        self.to(device)

    def change(self, row):
        """U and V for the day, from earlier days and the move of the forward."""
        days = row.unsqueeze(1) - torch.arange(DAYS, 0, -1, device=row.device)
        age = ((self.unix[row].unsqueeze(1) - self.unix[days]).double() / 86400).float()
        x = self.project(self.inputs[days]) + self.project_extra(self.features[row]).unsqueeze(1)
        out = self.generator(self.encoder(x, age))
        return out[:, RANK * EMBED:].view(-1, THETA, RANK), out[:, :RANK * EMBED].view(-1, RANK, EMBED)

    def weights(self, row, refs):
        U, V = self.change(row)
        e = self.encode(refs)
        return self.model._hyper.fc2(e) + torch.einsum('bpr,br->bp', U, torch.einsum('bre,be->br', V, e))

    def last_layer(self, row):
        """The day's last layer W0 + UV, worked out once before any surface is built."""
        U, V = self.change(row)
        return self.model._hyper.fc2.weight + (U @ V)[0]
