#!/usr/bin/env python3
"""Datasaurus Dozen calendar.

Each month shows one of the 12 real Datasaurus Dozen datasets (Matejka &
Fitzmaurice, 2017; data/datasaurus_dozen.csv, from the datasauRus package).
The dozen on its own has r = -0.06. To get a strong, identical correlation
every month without distorting the shapes, the *whole page* is the plot:
"stars" are added across the page -- a band along the page diagonal that
carries the correlation, plus a sprinkling of background stars -- and a tiny
final affine tweak makes mean(x), mean(y), sd(x), sd(y), Pearson r (hence the
regression line and R^2) exactly identical in all twelve months.

Plot units: the page is 110 x 85 units (1 unit = 0.1 inch).

Usage: python make_calendar.py [--year 2027] [--r 0.816] [--out out]
"""
import argparse
import calendar
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle


def tint(c, k=0.55):
    """Blend colour c towards white by k."""
    return tuple(v + (1 - v) * k for v in to_rgb(c))

W, H = 110.0, 85.0
SHAPE_BOX = (48.0, 21.0, 58.0)        # x0, y0, side of the square holding a dozen shape
N_BAND, N_BG = 2500, 150             # diagonal-ribbon stars, background stars
MIN_GAP = 4.5                        # keep stars this far from shape points

MONTHS = [  # (dataset, colour)
    ("dino", "#e63946"), ("star", "#d6336c"), ("bullseye", "#f77f00"),
    ("circle", "#e09f00"), ("x_shape", "#2a9d8f"), ("dots", "#3a86ff"),
    ("h_lines", "#7b2cbf"), ("v_lines", "#6a994e"), ("wide_lines", "#bc4749"),
    ("high_lines", "#118ab2"), ("slant_up", "#2b9348"), ("slant_down", "#8d5a97"),
]
GLYPH = {2: "♥", 6: "☀", 12: "❄"}      # others: star
STAR, STAR_COL, BG_COL = "★", "#f2b705", "#9aa7c7"


def load_dozen(path):
    d = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["dataset"]].append((float(row["x"]), float(row["y"])))
    return {k: np.array(v) for k, v in d.items()}


# --------------------------------------------------------------------------
def far_from_shape(px, py, sx, sy):
    d = np.hypot(px[:, None] - sx[None, :], py[:, None] - sy[None, :]).min(axis=1)
    return d >= MIN_GAP


def compose(pts, r_target, seed):
    """Real dozen shape + page-wide stars. Returns natural x, y, kind(0 shape, 1 band, 2 bg)."""
    x0, y0, side = SHAPE_BOX
    sx, sy = x0 + pts[:, 0] / 100 * side, y0 + pts[:, 1] / 100 * side
    rng = np.random.default_rng(1000 + seed)
    nrm = np.array([-H, W]) / np.hypot(W, H)

    # background stars: uniform over the page
    m = N_BG * 12
    bx, by = rng.uniform(2, W - 2, m), rng.uniform(2, H - 2, m)
    ok = far_from_shape(bx, by, sx, sy)
    bx, by = bx[ok][:N_BG], by[ok][:N_BG]
    assert len(bx) == N_BG

    # band stars along the page diagonal; spread (sigma) is bisected to hit r
    M = N_BAND * 16
    t, z = rng.uniform(0.01, 0.99, M), rng.normal(0, 1, M)

    def build(sigma):
        px = t * W + sigma * z * nrm[0]
        py = t * H + sigma * z * nrm[1]
        ok = (px > 1) & (px < W - 1) & (py > 1) & (py < H - 1) & far_from_shape(px, py, sx, sy)
        px, py = px[ok][:N_BAND], py[ok][:N_BAND]
        assert len(px) == N_BAND, "increase the candidate pool"
        return (np.concatenate([sx, px, bx]), np.concatenate([sy, py, by]),
                np.concatenate([np.zeros(len(sx)), np.ones(len(px)), np.full(len(bx), 2)]))

    a, b = 0.2, 45.0
    for _ in range(50):
        mid = (a + b) / 2
        x, y, _ = build(mid)
        a, b = (mid, b) if np.corrcoef(x, y)[0, 1] > r_target else (a, mid)
    return build((a + b) / 2)


def exact_fix(x, y, r_target, mean, sd):
    """Small affine tweak: exact means, SDs (ddof=1) and Pearson r."""
    n = len(x)
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    resid = zy - (np.sum(zx * zy) / (n - 1)) * zx
    resid = (resid - resid.mean()) / resid.std(ddof=1)
    zy2 = r_target * zx + np.sqrt(1 - r_target ** 2) * resid
    return mean[0] + sd[0] * zx, mean[1] + sd[1] * zy2


def stats(x, y):
    r = np.corrcoef(x, y)[0, 1]
    slope, icpt = np.polyfit(x, y, 1)
    return dict(mean_x=x.mean(), mean_y=y.mean(), sd_x=x.std(ddof=1),
                sd_y=y.std(ddof=1), r=r, r2=r * r, slope=slope, intercept=icpt)


# --------------------------------------------------------------------------
def draw_month(pdf, year, month, x, y, kind, name, st, colour, out_png):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#fbf8f1")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    for g in range(0, 111, 10):
        ax.plot([g, g], [0, H], color="#ece6d6", lw=0.5, zorder=0)
    for g in range(0, 86, 10):
        ax.plot([0, W], [g, g], color="#ece6d6", lw=0.5, zorder=0)

    ax.plot([0, W], [st["intercept"], st["intercept"] + st["slope"] * W],
            color="#222", lw=1.1, ls="--", zorder=2, alpha=0.7)

    glyph = GLYPH.get(month, STAR)
    for u, v in zip(x[kind == 1], y[kind == 1]):
        ax.text(u, v, glyph, color=STAR_COL if glyph == STAR else colour, fontsize=12,
                ha="center", va="center", fontfamily="DejaVu Sans", zorder=3, alpha=0.95)
    for u, v in zip(x[kind == 2], y[kind == 2]):
        ax.text(u, v, "✦", color=BG_COL, fontsize=7, ha="center", va="center",
                fontfamily="DejaVu Sans", zorder=1, alpha=0.8)
    ax.scatter(x[kind == 0], y[kind == 0], s=40, color=colour, edgecolor="#1b1b1b",
               linewidth=0.7, zorder=4)

    # calendar card
    ax.add_patch(FancyBboxPatch((3, 3), 41, 79, boxstyle="round,pad=0,rounding_size=2",
                                fc="white", ec="#d9d2bf", lw=0.8, alpha=0.74, zorder=5))
    ax.text(5, 77, calendar.month_name[month].upper(), fontsize=27, fontweight="bold",
            color=colour, va="center", zorder=6)
    ax.text(42, 77, str(year), fontsize=15, color="#666", ha="right", va="center", zorder=6)
    ax.text(5, 72.6, f"Datasaurus Dozen: “{name}”", fontsize=9.5, style="italic",
            color="#555", va="center", zorder=6)

    cal = calendar.Calendar(0).monthdayscalendar(year, month)
    cw, ch, gx, gy = 5.5, 6.6, 4.5, 68.0
    for i, d in enumerate(["M", "T", "W", "T", "F", "S", "S"]):
        ax.text(gx + i * cw + cw / 2, gy - 1.5, d, ha="center", va="center", fontsize=8,
                fontweight="bold", color=colour, zorder=6)
    for ri, week in enumerate(cal):
        for c, day in enumerate(week):
            if day == 0:
                continue
            xx, yy = gx + c * cw, gy - 3 - (ri + 1) * ch
            ax.add_patch(Rectangle((xx, yy), cw, ch, fc="#f1ece0" if c >= 5 else "none",
                                   ec="#c9c2ad", lw=0.6, zorder=6))
            ax.text(xx + 0.5, yy + ch - 0.6, str(day), fontsize=8.5, va="top", zorder=7,
                    color="#b00" if c == 6 else "#333")

    # stats
    sy0 = 22.0
    ax.text(5, sy0, "IDENTICAL IN EVERY MONTH", fontsize=7.5, fontweight="bold", color="#444",
            va="center", zorder=6)
    items = [("mean x", st["mean_x"]), ("mean y", st["mean_y"]), ("sd x", st["sd_x"]),
             ("sd y", st["sd_y"]), ("Pearson r", st["r"]), ("R²", st["r2"])]
    for i, (k, v) in enumerate(items):
        col, row = divmod(i, 3)
        xx, yy = 5 + col * 20, sy0 - 3.4 - row * 3.4
        ax.text(xx, yy, k, fontsize=8.5, color="#555", va="center", zorder=6)
        ax.text(xx + 17, yy, f"{v:.2f}", fontsize=8.5, color="#222", ha="right", va="center",
                fontfamily="monospace", fontweight="bold" if k == "R²" else "normal", zorder=6)
    ax.text(5, 5.2, f"n = {len(x)}: {int((kind == 0).sum())} dozen points + "
            f"{int((kind == 1).sum())} ribbon + {int((kind == 2).sum())} background stars",
            fontsize=6.3, color="#888", va="center", zorder=6)
    ax.text(W - 2, 2, "Same stats, different graphs · Matejka & Fitzmaurice (2017) · "
            "plot units: 1 = 0.1 in", fontsize=6.3, color="#888", ha="right", va="center", zorder=6)

    fig.savefig(out_png, dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--r", type=float, default=0.816,
                    help="shared Pearson r (0.816 = Anscombe, R²≈0.67)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "data", "datasaurus_dozen.csv"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dozen = load_dozen(a.data)

    natural = [compose(dozen[name], a.r, m) for m, (name, _) in enumerate(MONTHS, start=1)]
    mean = (np.mean([x.mean() for x, _, _ in natural]), np.mean([y.mean() for _, y, _ in natural]))
    sd = (np.mean([x.std(ddof=1) for x, _, _ in natural]), np.mean([y.std(ddof=1) for _, y, _ in natural]))

    rows, all_stats = [], []
    with PdfPages(os.path.join(a.out, f"datasaurus_calendar_{a.year}.pdf")) as pdf:
        for m, ((name, colour), (x, y, kind)) in enumerate(zip(MONTHS, natural), start=1):
            r_nat = np.corrcoef(x, y)[0, 1]
            x, y = exact_fix(x, y, a.r, mean, sd)
            st = stats(x, y)
            all_stats.append(st)
            off = int(((x < 0) | (x > W) | (y < 0) | (y > H)).sum())
            print(f"{calendar.month_name[m]:9s} {name:10s} natural r={r_nat:.3f}  off-page={off}")
            draw_month(pdf, a.year, m, x, y, kind, name, st, colour,
                       os.path.join(a.out, f"{m:02d}_{calendar.month_name[m].lower()}.png"))
            rows += [(calendar.month_name[m], name, f"{u:.6f}", f"{v:.6f}",
                      ("shape", "band_star", "background_star")[int(k)])
                     for u, v, k in zip(x, y, kind)]

    with open(os.path.join(a.out, "datasets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "dozen_dataset", "x", "y", "kind"])
        w.writerows(rows)

    print("max spread across the 12 months (should be ~0):")
    for k in all_stats[0]:
        vals = [s[k] for s in all_stats]
        print(f"  {k:10s} {vals[0]: .4f}   spread {max(vals) - min(vals):.2e}")


if __name__ == "__main__":
    main()
