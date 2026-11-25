"""Black-Scholes in forward form and implied volatility."""
import numpy as np
from scipy.special import ndtr


def price(strike, tau, forward, is_call, sigma, rate):
    """Option price; is_call is +1 for a call and -1 for a put."""
    k, t, f, c, v, r = np.broadcast_arrays(*[np.asarray(x, float) for x in (strike, tau, forward, is_call, sigma, rate)])
    st = v * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * st * st) / np.where(st > 0, st, 1.0)
    value = np.exp(-r * t) * c * (f * ndtr(c * d1) - k * ndtr(c * (d1 - st)))
    return np.where(st > 0, value, np.exp(-r * t) * np.maximum(c * (f - k), 0))


def delta(strike, tau, forward, is_call, sigma):
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * tau) / (sigma * np.sqrt(tau))
    return is_call * ndtr(is_call * d1)


def implied_volatility(option_price, strike, tau, forward, is_call, rate, high=5.0, steps=48):
    """Bisection on [0, high]; NaN where no volatility reproduces the price."""
    p, k, t, f, c, r = np.broadcast_arrays(*[np.asarray(x, float) for x in (option_price, strike, tau, forward, is_call, rate)])
    lo, hi = np.zeros(p.shape), np.full(p.shape, high)
    ok = (p > price(k, t, f, c, 0.0, r)) & (p <= price(k, t, f, c, high, r))
    for _ in range(steps):
        mid = (lo + hi) / 2
        low = price(k, t, f, c, mid, r) < p
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return np.where(ok, (lo + hi) / 2, np.nan)
