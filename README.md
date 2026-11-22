# Datasaurus / Anscombe calendar

Twelve months, twelve different scatter plots, one set of statistics: identical mean(x),
mean(y), sd(x), sd(y), Pearson r = 0.816 and therefore the same regression line and
R² = 0.67 (equal to ~1e-14).

* January is the real **Datasaurus** (`dino` from the Datasaurus Dozen, Matejka & Fitzmaurice 2017;
  data in `data/datasaurus_dozen.csv`, from the `datasauRus` CRAN package).
* The other months are my own drawings: Heart, Anscombe II/III/IV, Flower, Sun, Wave, Binary star,
  Spiral, X, Tree.
* **The main drawing is never distorted.** It is drawn at uniform scale. A baby copy of it
  (64 points) and 220 stars sit in the two corners along the diagonal and carry the correlation;
  the width of the star ribbon is tuned so r matches. One small affine map applied to those
  extras only (never to the main drawing) then makes the statistics exactly equal.
* ~490 points per plot (142 drawing + 128 baby + 220 stars).

```bash
pip install numpy matplotlib scipy
python make_calendar.py --year 2027      # -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.9          # different shared correlation (may need other settings)
```
