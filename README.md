# Datasaurus / Anscombe calendar

Twelve months, twelve different scatter plots.

* **January is the original Datasaurus**, untouched (`dino` from the Datasaurus Dozen, Matejka &
  Fitzmaurice 2017; `data/datasaurus_dozen.csv`, from the `datasauRus` CRAN package). Nothing is
  added or moved, so it keeps its own stats: r = -0.06, R² = 0.004.
* **February-December** share *identical* mean(x), mean(y), sd(x), sd(y), Pearson r = 0.816 and
  therefore the same regression line and R² = 0.67 (equal to ~1e-13). Drawings: Heart,
  Anscombe II/III/IV, Flower, Sun, Wave, Binary star, Spiral, X, Tree.
* **The main drawing is never distorted.** It is drawn at uniform scale. Around it:
  a baby copy of it in each corner (64 points each), 360 stars along the diagonal, and 130 small
  sparse points that fill the white gaps (farthest-point sampling). The stars carry the correlation
  (the ribbon width is tuned so r matches); one small affine map applied to those extras only
  (never to the main drawing) then makes the statistics exactly equal.
* ~760 points per plot (142 drawing + 128 baby + 360 stars + 130 sparse).

```bash
pip install numpy matplotlib scipy
python make_calendar.py --year 2027      # ~40 s -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
```
