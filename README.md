# Datasaurus calendar

A 12-month calendar where each month's plot is a different shape, but every dataset
has the same mean, SD, Pearson r, regression line and R² (equal to floating-point precision).

```bash
pip install numpy matplotlib
python make_calendar.py --year 2027          # -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.816            # Anscombe-style R² (~0.67); shapes get sheared
```

Each shape is generated, then an affine "stats fix" sets the exact means, SDs and r.
Default r = -0.06 (Datasaurus), so R² = 0.0036 and shapes stay recognisable.
