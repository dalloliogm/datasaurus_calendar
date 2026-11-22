# Datasaurus Dozen calendar

A 12-month calendar. Each month shows one of the 12 real **Datasaurus Dozen** datasets
(Matejka & Fitzmaurice, *Same Stats, Different Graphs*, 2017; dino, star, bullseye, circle,
x_shape, dots, h_lines, v_lines, wide_lines, high_lines, slant_up, slant_down; the 13th, `away`,
is left out). Data: `data/datasaurus_dozen.csv`, taken from the `datasauRus` package on CRAN.

The dozen alone has r ≈ -0.06. To get an identical, strong correlation (default r = 0.816,
R² = 0.67) in every month **without distorting the shapes**, the whole page is the plot:
~2,500 "stars" form a wide ribbon along the page diagonal and 150 more are sprinkled on
the background. Stars keep clear of the dozen's points. A tiny final affine tweak makes
mean(x), mean(y), sd(x), sd(y), Pearson r (and so the regression line and R²) exactly identical
in all twelve months (verified to ~1e-14).

```bash
pip install numpy matplotlib
python make_calendar.py --year 2027     # ~1.5 min -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.9         # different shared correlation
```

`out/datasets.csv` lists every point with its month, dozen dataset and kind
(`shape`, `band_star`, `background_star`). Plot units: the page is 110 x 85 units (1 unit = 0.1 in).
