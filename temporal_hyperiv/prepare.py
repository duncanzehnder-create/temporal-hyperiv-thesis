"""OptionsDX end-of-day files to cleaned SPX implied-volatility surfaces."""
import json
import tempfile
from pathlib import Path

import exchange_calendars
import numpy as np
import pandas as pd
import py7zr

from hyperiv.data_util import find_closest_elements
from temporal_hyperiv.finance import delta, implied_volatility, price

COLUMNS = ['log_moneyness', 'tau', 'implied_volatility', 'strike', 'forward', 'rate', 'price', 'bid', 'ask', 'is_call']
TENORS = {'DGS1MO': 1 / 12, 'DGS3MO': 0.25, 'DGS6MO': 0.5, 'DGS1': 1.0, 'DGS2': 2.0}
NUMERIC = ['QUOTE_UNIXTIME', 'STRIKE', 'UNDERLYING_LAST', 'DTE', 'C_BID', 'C_ASK', 'P_BID', 'P_ASK']
MAX_TAU, MIN_PRICE, MAX_SPREAD, REPRICING_TOLERANCE = 2, 0.1, 0.5, 0.01
PARITY_PAIRS, MAX_PARITY_DISTANCE = 3, 0.1
MATURITIES, DELTAS = [7, 30, 90], [0.5, 0.25, -0.25]


def rate_curve(rates, day):
    """Treasury yields of the last earlier date, as a function of maturity."""
    earlier = rates[rates.observation_date < pd.Timestamp(day)].iloc[-1]
    if (pd.Timestamp(day) - earlier.observation_date).days > 7:
        raise ValueError(f'No recent rate for {day}')
    values = earlier[list(TENORS)].to_numpy(float) / 100
    known = np.isfinite(values)
    return lambda tau: float(np.interp(tau, np.array(list(TENORS.values()))[known], values[known]))


def parity_forward(group, rate):
    """Forward from the three call-put pairs nearest the index, and the rows used."""
    if len(group) <= PARITY_PAIRS:
        return None
    near = group.assign(distance=abs(np.log(group.STRIKE / group.UNDERLYING_LAST)))
    near = near.sort_values(['distance', 'STRIKE'], kind='stable').head(PARITY_PAIRS)
    sides = near[['C_BID', 'C_ASK', 'P_BID', 'P_ASK']].to_numpy()
    if near.distance.max() > MAX_PARITY_DISTANCE or not np.isfinite(sides).all() or (sides <= 0).any():
        return None
    if (sides[:, 0] > sides[:, 1]).any() or (sides[:, 2] > sides[:, 3]).any():
        return None
    growth = np.exp(rate * float(near.DTE.iloc[0]) / 365)
    low = float((near.STRIKE + growth * (near.C_BID - near.P_ASK)).max())
    high = float((near.STRIKE + growth * (near.C_ASK - near.P_BID)).min())
    if low > high or high <= 0:
        return None
    middle = near.STRIKE + growth * ((near.C_BID + near.C_ASK - near.P_BID - near.P_ASK) / 2)
    forward = float(np.clip(np.median(middle), low, high))
    return (forward, near.index) if forward > 0 else None


def clean_day(raw, curve):
    """One day's out-of-the-money quotes with is_ref marking the nine reference quotes, or None."""
    parts = []
    for _, group in raw.groupby('EXPIRE_DATE', sort=True):
        tau = float(group.DTE.iloc[0]) / 365
        if not 0 < tau <= MAX_TAU:
            continue
        rate = curve(tau)
        found = parity_forward(group, rate)
        if found is None:
            continue
        forward, used = found
        rest = group.drop(index=used)
        for side, is_call in (('C', 1), ('P', -1)):
            bid, ask = rest[f'{side}_BID'], rest[f'{side}_ASK']
            mid = (bid + ask) / 2
            keep = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid) & (mid >= MIN_PRICE)
            keep &= (ask - bid <= MAX_SPREAD * mid) & (is_call * (rest.STRIKE - forward) >= 0)
            parts.append(pd.DataFrame({'strike': rest.STRIKE[keep], 'days': rest.DTE[keep], 'tau': tau, 'is_call': is_call,
                                       'bid': bid[keep], 'ask': ask[keep], 'price': mid[keep], 'forward': forward, 'rate': rate}))
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    q = pd.concat(parts, ignore_index=True)
    q['log_moneyness'] = np.log(q.strike / q.forward)
    q['implied_volatility'] = implied_volatility(q.price, q.strike, q.tau, q.forward, q.is_call, q.rate)
    q = q[q.implied_volatility > 0]
    error = abs(price(q.strike, q.tau, q.forward, q.is_call, q.implied_volatility, q.rate) - q.price)
    q = q[error <= REPRICING_TOLERANCE]
    q = q.sort_values(['days', 'strike', 'is_call'], kind='stable').reset_index(drop=True)
    q['delta'] = delta(q.strike, q.tau, q.forward, q.is_call, q.implied_volatility)
    if q.days.nunique() < 3:
        return None
    refs = []
    for _, expiry in find_closest_elements(q, 'days', MATURITIES).groupby('days', sort=True):
        if expiry.delta.nunique() >= 3:
            refs.extend(find_closest_elements(expiry, 'delta', DELTAS).index)
    if len(refs) != 9 or len(q) <= 9:
        return None
    return q.assign(is_ref=q.index.isin(refs))


def read_days(archive):
    """(date, raw quotes) for every day in one OptionsDX archive."""
    with tempfile.TemporaryDirectory() as folder:
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(folder)
        for path in sorted(Path(folder).rglob('*.txt')):
            raw = pd.read_csv(path, skipinitialspace=True, low_memory=False)
            raw.columns = [c.strip().strip('[]') for c in raw.columns]
            for c in ('QUOTE_DATE', 'EXPIRE_DATE'):
                raw[c] = raw[c].astype(str).str.strip()
            raw[NUMERIC] = raw[NUMERIC].apply(pd.to_numeric, errors='coerce')
            yield from raw.groupby('QUOTE_DATE', sort=True)


def prepare(archives, rates_csv, output):
    """Write quotes.npy (all but the reference quotes) and surfaces.json (one entry per day)."""
    rates = pd.read_csv(rates_csv, parse_dates=['observation_date']).sort_values('observation_date')
    rates[list(TENORS)] = rates[list(TENORS)].apply(pd.to_numeric, errors='coerce')
    rates = rates.dropna(subset=list(TENORS), how='all')
    sessions = set(exchange_calendars.get_calendar('XNYS', start='2010-01-01', end='2023-12-31').sessions.strftime('%Y-%m-%d'))
    days = {}
    for archive in archives:
        print(Path(archive).name, flush=True)
        for date, raw in read_days(archive):
            if date in sessions and date not in days:
                q = clean_day(raw, rate_curve(rates, date))
                if q is not None:
                    days[date] = q, int(raw.QUOTE_UNIXTIME.iloc[0])
    surfaces, blocks, start = [], [], 0
    for date in sorted(days):
        q, unix = days[date]
        blocks.append(q.loc[~q.is_ref, COLUMNS].to_numpy(float))
        surfaces.append({'date': date, 'unix': unix, 'start': start, 'stop': start + len(blocks[-1]),
                         'refs': q.loc[q.is_ref, COLUMNS[:3]].values.tolist()})
        start += len(blocks[-1])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'quotes.npy', np.concatenate(blocks))
    (output / 'surfaces.json').write_text(json.dumps(surfaces))
    return len(surfaces), start
