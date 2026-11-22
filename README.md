# Datasaurus / Anscombe calendar

Twelve months, twelve different scatter plots, one set of statistics: identical mean(x),
mean(y), sd(x), sd(y), Pearson r = 0.816 and therefore the same regression line and
R² = 0.67 (equal to ~1e-14).

* January is the real **Datasaurus** (`dino` from the Datasaurus Dozen, Matejka & Fitzmaurice 2017;
  data in `data/datasaurus_dozen.csv`, from the `datasauRus` CRAN package).
* The other months are my own shapes: Heart, Anscombe II/III/IV, Flower, Sun, Wave, Binary star,
  Spiral, X, Tree.
* Each plot = the shape (142 points, drawn at uniform scale) + 40 "stars" trailing along the
  diagonal. The stars carry most of the correlation, so the shape is barely deformed; a tiny
  affine tweak then makes the statistics exactly equal.

```bash
pip install numpy matplotlib
python make_calendar.py --year 2027      # -> out/datasaurus_calendar_2027.pdf, PNGs, datasets.csv
python make_calendar.py --r 0.9          # different shared correlation
```
