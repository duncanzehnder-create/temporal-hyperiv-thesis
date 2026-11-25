"""Build the surfaces from OptionsDX archives and FRED Treasury yields."""
import argparse
from pathlib import Path

from temporal_hyperiv.prepare import prepare

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archives', type=Path, help='folder with the spx_eod_*.7z files')
p.add_argument('rates', type=Path, help='CSV with observation_date, DGS1MO, DGS3MO, DGS6MO, DGS1 and DGS2')
p.add_argument('--output', type=Path, default='data')
a = p.parse_args()
print('%d days, %d quotes' % prepare(sorted(a.archives.glob('spx_eod_*.7z')), a.rates, a.output))
