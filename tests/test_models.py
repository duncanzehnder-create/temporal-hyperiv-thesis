import numpy as np
import pytest
import torch

from temporal_hyperiv.baseline import HyperIV
from temporal_hyperiv.carry import arbitrage_step, carry_table, plain_step
from temporal_hyperiv.data import Batches
from temporal_hyperiv.fields import CELLS, cell_means, cells, error_fields
from temporal_hyperiv.finance import implied_volatility, price
from temporal_hyperiv.gru import G5


class FakeMarket:
    def __init__(self, days=8, quotes=200):
        rng = np.random.default_rng(0)
        self.dates = [f'2020-01-{d + 1:02d}' for d in range(days)]
        self.unix = np.arange(days) * 86400
        k, tau = rng.uniform(-0.5, 0.2, (days, quotes)), rng.uniform(0.02, 1.5, (days, quotes))
        iv = 0.2 - 0.2 * k + 0.02 * rng.normal(size=k.shape)
        self.panels = np.stack([k, tau, iv] + [np.full(k.shape, 100.0)] * 7, axis=-1)
        self.refs = self.panels[:, :9, :3]

    def panel(self, row):
        return self.panels[row]


@pytest.fixture
def market():
    return FakeMarket()


@pytest.fixture
def base():
    torch.manual_seed(0)
    return HyperIV().eval()


def test_implied_volatility_inverts_price():
    k, sigma = np.linspace(80, 120, 9), np.linspace(0.1, 0.5, 9)
    for is_call in (1, -1):
        p = price(k, 0.5, 100, is_call, sigma, 0.03)
        assert np.allclose(implied_volatility(p, k, 0.5, 100, is_call, 0.03), sigma, atol=1e-8)
    parity = price(k, 0.5, 100, 1, sigma, 0.03) - price(k, 0.5, 100, -1, sigma, 0.03)
    assert np.allclose(parity, np.exp(-0.015) * (100 - k))


def test_network_sizes(base):
    assert sum(p.numel() for p in base.parameters()) == 243153
    assert base.weights(None, torch.rand(1, 9, 3)).shape == (1, 337)


def test_cells():
    assert CELLS == 80 and cells(-1.0, 0.001) == 0 and cells(1.0, 2.0) == 79
    count, mean = cell_means(np.zeros(4), np.full(4, 0.1), np.arange(4.0))
    assert count.sum() == 4 and mean[cells(0.0, 0.1)] == 1.5


def test_carry_moves_towards_the_quotes(market, base):
    table = carry_table(base, market, [1], 'cpu')
    p = torch.tensor(market.panel(0)[None, :, :3], dtype=torch.float32)
    theta = base.weights(None, torch.tensor(market.refs[:1], dtype=torch.float32))
    error = lambda th: float((base.decode(th, p[..., :2]) - p[..., 2]).abs().mean())
    assert error(theta + table[1]) < error(theta)
    assert torch.equal(table[0], torch.zeros(337))


def test_steps_agree_without_penalty(market, base):
    plain = carry_table(base, market, [1], 'cpu', plain_step)
    free = carry_table(base, market, [1], 'cpu', arbitrage_step, penalty=0.0)
    assert torch.allclose(plain, free, atol=1e-4)


def test_carry_ignores_today(market, base):
    before = carry_table(base, market, [3], 'cpu')
    market.panels[3:] += 1
    assert torch.equal(before, carry_table(base, market, [3], 'cpu'))


def test_g5_starts_at_its_base(market, base):
    data = Batches(market, [6, 7], 'cpu')
    model = G5(base, market, *error_fields(base, market, 7, 'cpu', range(8)), 'cpu')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 78548
    assert torch.allclose(model.weights(data.row, data.refs), base.weights(None, data.refs), atol=1e-6)


def test_g5_ignores_todays_withheld_quotes(market, base):
    changed = FakeMarket()
    changed.panels[6, :, 2] += 1
    models = [G5(base, m, *error_fields(base, m, 6, 'cpu', range(6)), 'cpu') for m in (market, changed)]
    torch.nn.init.normal_(models[0].generator[-1].weight, std=0.02)
    models[1].load_state_dict(models[0].state_dict())
    row = torch.tensor([6])
    assert torch.equal(models[0].last_layer(row), models[1].last_layer(row))


def test_last_layer_gives_the_same_weights(market, base):
    data = Batches(market, [7], 'cpu')
    model = G5(base, market, *error_fields(base, market, 7, 'cpu', range(8)), 'cpu')
    torch.nn.init.normal_(model.generator[-1].weight, std=0.02)
    once = torch.nn.functional.linear(model.encode(data.refs), model.last_layer(data.row), base.model._hyper.fc2.bias)
    assert torch.allclose(once, model.weights(data.row, data.refs), atol=1e-5)
