#!/usr/bin/env python3
"""Datasaurus / Anscombe-inspired calendar.

Every month shows a different scatter-plot shape, but all twelve datasets have
*exactly* the same mean(x), mean(y), sd(x), sd(y), Pearson r and therefore the
same linear-regression line and R^2.

How the stats are matched: each shape is generated as a point cloud, then an
affine "stats fix" is applied (standardise, re-mix the residual so that the
correlation equals TARGET_R, rescale to the target means/SDs). Because the fix
is affine, the shape stays recognisable (it is only sheared/stretched), and the
statistics are equal to floating-point precision rather than "close enough".

Usage:  python make_calendar.py [--year 2027] [--r 0.816] [--out out]
"""
import argparse
import calendar
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

N = 120
TARGET_MEAN = (50.0, 50.0)
TARGET_SD = (20.0, 20.0)       # equal SDs so the diagonal stretch is symmetric


# --------------------------------------------------------------------------
# Shapes. Each returns (x, y) arrays of N points in arbitrary units.
# --------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def along_path(px, py, n=N, jitter=0.0, seed=0):
    """n points spread evenly (by arc length) along a polyline."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    x, y = np.interp(t, s, px), np.interp(t, s, py)
    r = _rng(seed)
    return x + r.normal(0, jitter, n), y + r.normal(0, jitter, n)


def linear():                      # Jan  - Anscombe I (noisy blob)
    r = _rng(1)
    return r.normal(0, 1, N), r.normal(0, 1, N)


def parabola():                    # Mar  - Anscombe II
    r = _rng(2)
    x = np.sort(r.uniform(0, 10, N))
    return x, -0.13 * (x - 5) ** 2 + 9 + r.normal(0, 0.12, N)


def outlier_line():                # Apr  - Anscombe III (flat line + outlier)
    r = _rng(3)
    k = 8                                  # a few outliers keep the plot readable
    x = np.sort(r.uniform(0, 10, N - k))
    y = 5 + r.normal(0, 0.05, N - k)
    return (np.append(x, r.uniform(8.5, 9.5, k)),
            np.append(y, r.uniform(8, 13, k)))


def vertical_outlier():            # Nov  - Anscombe IV
    r = _rng(4)
    k = 16                                 # a second, smaller column
    x = np.full(N - k, 4.0) + r.normal(0, 0.02, N - k)
    y = r.uniform(3, 9, N - k)
    return (np.append(x, 8 + r.normal(0, 0.02, k)),
            np.append(y, r.uniform(5, 9, k)))


def heart():                       # Feb
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x + _rng(5).normal(0, .15, N), y + _rng(6).normal(0, .15, N)


def flower():                      # May  (rose curve, 5 petals)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rad = np.abs(np.cos(2.5 * t)) + 0.15
    return rad * np.cos(t), rad * np.sin(t)


def ring():                        # Jun  (sun / circle)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.cos(t) + _rng(7).normal(0, .02, N), np.sin(t) + _rng(8).normal(0, .02, N)


def sine():                        # Jul
    x = np.linspace(0, 4 * np.pi, N)
    return x, np.sin(x) * 2 + 0.6 * x


def clusters():                    # Aug
    r = _rng(9)
    h = N // 2
    return (np.concatenate([r.normal(2, .8, h), r.normal(8, .8, N - h)]),
            np.concatenate([r.normal(2, .8, h), r.normal(8, .8, N - h)]))


def spiral():                      # Sep
    t = np.linspace(0.3, 4.2 * np.pi, N)
    return t * np.cos(t), t * np.sin(t)


def x_shape():                     # Oct
    return along_path([0, 10, 5, 0, 10], [0, 10, 5, 10, 0], jitter=0.12, seed=10)


def tree():                        # Dec  (zig-zag Christmas tree + trunk)
    px = [5, 1, 3, 0, 3.5, 0.5, 4.5, 4.5, 5.5, 5.5, 4.5,
          9.5, 6.5, 10, 7, 10.5, 6.5, 9.5, 5.5, 5.5]
    py = [14, 10, 10, 6.5, 6.5, 3, 3, 0, 0, 3, 3,
          3, 6.5, 6.5, 10, 10, 10, 10, 14, 14]
    # simple, clean silhouette instead of the scribble above
    px = [5, 1.5, 3.5, 0, 3, -0.5, 4.2, 4.2, 5.8, 5.8, 10.5, 7, 10, 6.5, 8.5, 5]
    py = [14, 10, 10, 6.5, 6.5, 3, 3, 0, 0, 3, 3, 6.5, 6.5, 10, 10, 14]
    return along_path(px, py, jitter=0.05, seed=11)


# month -> (shape fn, caption)
SHAPES = [
    (linear, "Anscombe I (the humble blob)"),
    (heart, "Heart"),
    (parabola, "Anscombe II"),
    (outlier_line, "Anscombe III"),
    (flower, "Flower"),
    (ring, "Sun"),
    (sine, "Wave"),
    (clusters, "Binary star"),
    (spiral, "Spiral"),
    (x_shape, "X marks the spot"),
    (vertical_outlier, "Anscombe IV"),
    (tree, "Tree"),
]


# --------------------------------------------------------------------------
def match_stats(x, y, r_target, mean=TARGET_MEAN, sd=TARGET_SD):
    """Exact means, SDs and Pearson r, by stretching the shape along the diagonal.

    The shape's own axes (s, t) are whitened, then mapped so that s runs along
    the y = x diagonal and t across it:  zx = p s + q t,  zy = p s - q t with
    p = sqrt((1+r)/2), q = sqrt((1-r)/2).  That gives var = 1 for both and
    corr = r, and keeps right angles of the shape intact up to a uniform
    stretch (a vertical shear would distort it far more).
    """
    s = (x - x.mean()) / x.std()
    t = (y - y.mean()) / y.std()
    t = t - np.mean(s * t) * s
    t = (t - t.mean()) / t.std()
    p, q = np.sqrt((1 + r_target) / 2), np.sqrt((1 - r_target) / 2)
    zx, zy = p * s + q * t, p * s - q * t
    return mean[0] + sd[0] * zx, mean[1] + sd[1] * zy


def stats(x, y):
    r = np.corrcoef(x, y)[0, 1]
    slope, icpt = np.polyfit(x, y, 1)
    return dict(mean_x=x.mean(), mean_y=y.mean(), sd_x=x.std(ddof=1),
                sd_y=y.std(ddof=1), r=r, r2=r * r, slope=slope, intercept=icpt)


# --------------------------------------------------------------------------
PALETTE = ["#e63946", "#d6336c", "#f77f00", "#e09f00", "#2a9d8f", "#fcbf49",
           "#3a86ff", "#7b2cbf", "#6a994e", "#bc4749", "#118ab2", "#2b9348"]


def draw_month(pdf, year, month, x, y, caption, st, colour, out_png):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#fbf8f1")

    # title
    fig.text(0.06, 0.94, calendar.month_name[month].upper(), fontsize=34,
             fontweight="bold", color=colour, va="center")
    fig.text(0.94, 0.94, str(year), fontsize=34, color="#555", ha="right", va="center")
    fig.text(0.06, 0.895, f"“{caption}” — same mean, SD, correlation and R² as every other month",
             fontsize=10.5, color="#555", style="italic")

    # scatter plot
    ax = fig.add_axes([0.06, 0.46, 0.42, 0.42])
    ax.set_facecolor("white")
    px, py = 0.08 * np.ptp(x), 0.08 * np.ptp(y)
    ax.scatter(x, y, s=22, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, st["slope"] * xs + st["intercept"], color="#222", lw=1.2, ls="--", zorder=2)
    lo = min(0, x.min() - 3, y.min() - 3)
    hi = max(100, x.max() + 3, y.max() + 3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(color="#eee", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors="#888")

    # stats panel
    pn = fig.add_axes([0.62, 0.46, 0.32, 0.42])
    pn.axis("off")
    rows = [("mean x", st["mean_x"]), ("mean y", st["mean_y"]),
            ("sd x", st["sd_x"]), ("sd y", st["sd_y"]),
            ("Pearson r", st["r"]), ("R²", st["r2"]),
            ("slope", st["slope"]), ("intercept", st["intercept"])]
    pn.text(0, 1, "IDENTICAL EVERY MONTH", fontsize=9, fontweight="bold", color="#444", va="top")
    for i, (k, v) in enumerate(rows):
        yy = 0.86 - i * 0.105
        pn.text(0, yy, k, fontsize=11, color="#555", va="center")
        pn.text(1, yy, f"{v:.2f}", fontsize=11, color="#222", ha="right", va="center",
                fontfamily="monospace", fontweight="bold" if k == "R²" else "normal")
    pn.text(0, -0.02, f"n = {len(x)} points", fontsize=9, color="#888", va="center")

    # day grid
    cal = calendar.Calendar(firstweekday=0).monthdayscalendar(year, month)
    g = fig.add_axes([0.06, 0.04, 0.88, 0.37])
    g.set_xlim(0, 7)
    g.set_ylim(len(cal) + 1, 0)
    g.axis("off")
    for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        g.text(i + 0.5, 0.5, d, ha="center", va="center", fontsize=10,
               fontweight="bold", color=colour)
    for r_i, week in enumerate(cal, start=1):
        for c, day in enumerate(week):
            if day == 0:
                continue
            weekend = c >= 5
            g.add_patch(plt.Rectangle((c, r_i), 1, 1, fill=weekend,
                                      facecolor="#f1ece0", edgecolor="#bbb", lw=0.7))
            g.text(c + 0.08, r_i + 0.1, str(day), fontsize=11, va="top",
                   color="#b00" if c == 6 else "#333")

    fig.savefig(out_png, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--r", type=float, default=0.816,
                    help="shared Pearson r (0.816 = Anscombe, R²≈0.67; -0.06 = Datasaurus, R²≈0)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, all_stats = [], []
    with PdfPages(os.path.join(a.out, f"datasaurus_calendar_{a.year}.pdf")) as pdf:
        for m, ((fn, caption), colour) in enumerate(zip(SHAPES, PALETTE), start=1):
            x, y = fn()
            x, y = match_stats(np.asarray(x, float), np.asarray(y, float), a.r)
            st = stats(x, y)
            all_stats.append(st)
            draw_month(pdf, a.year, m, x, y, caption, st, colour,
                       os.path.join(a.out, f"{m:02d}_{calendar.month_name[m].lower()}.png"))
            rows += [(calendar.month_name[m], f"{u:.6f}", f"{v:.6f}") for u, v in zip(x, y)]

    with open(os.path.join(a.out, "datasets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "x", "y"])
        w.writerows(rows)

    keys = all_stats[0].keys()
    spread = {k: max(s[k] for s in all_stats) - min(s[k] for s in all_stats) for k in keys}
    print("max spread across the 12 months (should be ~0):")
    for k, v in spread.items():
        print(f"  {k:10s} {all_stats[0][k]: .4f}   spread {v:.2e}")


if __name__ == "__main__":
    main()
