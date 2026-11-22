# Datasaurus calendar

A 12-month calendar where each month's plot is a different shape, but every dataset
has the same mean, SD, Pearson r, regression line and R² (equal to floating-point precision).

```bash
pip install numpy matplotlib
python make_calendar.py --year 2027          # -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.816            # default: Anscombe-style R² (~0.67)
```

Each shape is generated, then stretched symmetrically along the y = x diagonal so that
means, SDs and Pearson r match exactly. Default r = 0.816 (Anscombe), so R² = 0.67 and
every month has r > 0.8. `--r -0.06` gives the near-zero Datasaurus-style correlation.
