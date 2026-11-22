# Datasaurus calendar

A 12-month calendar where each month's plot is a different shape, but every dataset
has the same mean, SD, Pearson r, regression line and R² (equal to floating-point precision).

```bash
pip install numpy matplotlib
python make_calendar.py --year 2027          # -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.816            # default: Anscombe-style R² (~0.67)
```

Each month = a shape drawn at uniform scale (no distortion) + 60 "stars" trailing along the
y = x diagonal on both sides (hearts in Feb, flowers in May, snowflakes in Dec). The star
tails carry most of the correlation (tail length is bisected to hit r), and a tiny affine
fix then makes means, SDs and Pearson r exactly identical in every month.
Default r = 0.816 (Anscombe, R² = 0.67, so r > 0.8). `--r -0.06` gives Datasaurus-style r.
`datasets.csv` marks each point as `shape` or `star`.
