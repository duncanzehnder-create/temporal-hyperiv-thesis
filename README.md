# Temporal HyperIV

Code for a MATH5925 thesis at UNSW Sydney. It extends [HyperIV](https://github.com/qmfin/hyperiv), which builds an implied-volatility surface from nine option quotes, with the errors of earlier days.

```
hyperiv/            original HyperIV code
temporal_hyperiv/   data, baseline, weight-space carry, G5, evaluation
scripts/            prepare, train, evaluate
results/            tables of the thesis
```

```bash
pip install -e .
python scripts/prepare_data.py ARCHIVES RATES
python scripts/train.py baseline --year 2022 --seed 17
python scripts/train.py g5 --year 2022 --seed 17
python scripts/evaluate.py --year 2022
```

The OptionsDX quotes are not included.
