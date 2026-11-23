#!/usr/bin/env python3
"""Datasaurus / Anscombe-inspired calendar.

Every month shows a different scatter-plot drawing, but all twelve datasets have
*exactly* the same mean(x), mean(y), sd(x), sd(y), Pearson r and therefore the
same regression line and R^2.

How: the main drawing is placed at uniform scale and is NEVER stretched or sheared.
A baby copy of it plus loose stars sit in the two corners along the diagonal; they
carry the correlation (the star ribbon width is bisected until r hits the target).
One small affine map applied to those extras only then makes the five statistics
exactly equal in all twelve months.

January is the real Datasaurus (data/datasaurus_dozen.csv, Matejka & Fitzmaurice 2017).

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
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages

N = 142                      # points per drawing (= size of the real Datasaurus dozen sets)


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


def dino():                        # Jan  - the real Datasaurus (Matejka & Fitzmaurice 2017)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "datasaurus_dozen.csv")
    with open(path) as f:
        pts = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f) if r["dataset"] == "dino"]
    a = np.array(pts)
    return a[:, 0], a[:, 1]


def parabola():                    # Mar  - Anscombe II
    r = _rng(2)
    x = np.sort(r.uniform(0, 10, N))
    return x, -0.13 * (x - 5) ** 2 + 9 + r.normal(0, 0.12, N)


def outlier_line():                # Apr  - Anscombe III (flat line + outlier)
    r = _rng(3)
    k = 8                                  # a few outliers keep the plot readable
    x = np.sort(r.uniform(0, 10, N - k))
    y = 5 + r.normal(0, 0.05, N - k)
    return (np.append(x, r.uniform(4, 6, k)),
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
    return x, np.sin(x) * 2


def clusters():                    # Aug
    r = _rng(9)
    h = N // 2
    return (np.concatenate([r.normal(2, .8, h), r.normal(8, .8, N - h)]),
            np.concatenate([r.normal(5, .8, h), r.normal(5, .8, N - h)]))


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
    (dino, "Datasaurus"),
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
TARGET_SD = 30.0             # common sd(x) = sd(y) of every month (plot units)
N_STARS = 360                # loose stars sprinkled along the diagonal tails
# baby copies of the month's drawing, per side of the diagonal: (scale, points, distance, sideways offset)
BABIES = [(0.44, 64, 54.0, 0.0)]
SHAPE_RADIUS = 30.0          # 95th-percentile radius of the main drawing (plot units)
RADIUS = {"Sun": 26.0}                  # optional per-caption overrides
TAIL_END = 82.0              # the star tails run out to this distance along the diagonal
GAP = 3.5                    # keep stars this far from drawings
STAR_GLYPH = {2: ("\u2665", "#ff7b9c"), 5: ("\u273f", "#f08fb5"), 12: ("\u2744", "#5aa9e6")}
DEFAULT_STAR = ("\u2605", "#f2b705")
SHAPE, MINI, STAR, SPARSE = 0, 1, 2, 3
N_SPARSE = 130               # small points that fill the white gaps (farthest-point sampling)
_g = np.linspace(3, 97, 36)
CAND_X, CAND_Y = [a.ravel() for a in np.meshgrid(_g, _g)]


def sparse_fill(ox, oy, k=None):
    """Pick k points from a grid, each time the one farthest from everything so far:
    an even, blue-noise-like sprinkle that lands in the biggest empty spots first."""
    k = N_SPARSE if k is None else k
    d = np.hypot(CAND_X[:, None] - ox[None, :], CAND_Y[:, None] - oy[None, :]).min(axis=1)
    pick = []
    for _ in range(k):
        i = int(np.argmax(d))
        pick.append(i)
        d = np.minimum(d, np.hypot(CAND_X - CAND_X[i], CAND_Y - CAND_Y[i]))
    return CAND_X[pick], CAND_Y[pick]


def _near(px, py, ox, oy, gap):
    d = np.hypot(px[:, None] - ox[None, :], py[:, None] - oy[None, :]).min(axis=1)
    return d < gap


def layout(fn, r_target, seed, radius=SHAPE_RADIUS):
    """Main drawing (untouched, uniform scale) + baby copies + stars on the diagonal.

    The babies and stars carry the correlation, so the main drawing never has to be
    stretched or sheared. The length of the star tails is bisected so r ~ r_target.
    Returns x, y, kind (0 drawing, 1 baby, 2 star).
    """
    x, y = (np.asarray(v, float) for v in fn())
    dx, dy = x - x.mean(), y - y.mean()
    k = radius / np.percentile(np.hypot(dx, dy), 95)
    sx, sy = 50 + dx * k, 50 + dy * k
    n = len(sx)

    rng = np.random.default_rng(100 + seed)
    shape_u = np.percentile(np.abs((sx - 50 + sy - 50) / np.sqrt(2)), 98)
    baby_pts = []                                   # (dx, dy, u, p) for each baby
    for scale, npts, u, p in BABIES:
        idx = rng.choice(n, npts, replace=False)
        baby_pts.append(((sx[idx] - 50) * scale, (sy[idx] - 50) * scale, u, p))

    M = N_STARS * 30
    t, side, perp1 = rng.uniform(0, 1, M), rng.choice([-1, 1], M), rng.normal(0, 1.0, M)
    lo = shape_u + 4

    def build(sigma):
        """Babies sit on the diagonal; stars fill both tails with sideways spread sigma."""
        bxs, bys = [], []
        for mx, my, u, p in baby_pts:
            for sgn in (1, -1):                      # point-mirrored on both sides
                bxs.append(50 + sgn * (u - p) / np.sqrt(2) + mx)
                bys.append(50 + sgn * (u + p) / np.sqrt(2) + my)
        bx, by = np.concatenate(bxs), np.concatenate(bys)
        ox, oy = np.concatenate([sx, bx]), np.concatenate([sy, by])
        u = side * (lo + t * (TAIL_END - lo))
        perp = sigma * perp1
        px, py = 50 + (u - perp) / np.sqrt(2), 50 + (u + perp) / np.sqrt(2)
        ok = ~_near(px, py, ox, oy, GAP) & (px > -6) & (px < 106) & (py > -6) & (py < 106)
        px, py = px[ok][:N_STARS], py[ok][:N_STARS]
        if len(px) < N_STARS:
            return None
        ax_, ay_ = np.concatenate([ox, px]), np.concatenate([oy, py])
        qx, qy = sparse_fill(ax_, ay_)
        kind = np.concatenate([np.full(n, SHAPE), np.full(len(bx), MINI),
                               np.full(N_STARS, STAR), np.full(N_SPARSE, SPARSE)])
        return np.concatenate([ax_, qx]), np.concatenate([ay_, qy]), kind

    def r_of(sigma):
        out = build(sigma)
        return None if out is None else np.corrcoef(out[0], out[1])[0, 1]

    a, b = 0.3, 40.0                                 # wider ribbon -> lower r
    for _ in range(26):
        mid = (a + b) / 2
        r = r_of(mid)
        a, b = (mid, b) if (r is not None and r > r_target) else (a, mid)
    out = build(a)
    return out if out is not None else build(b)


def fix_extras(x, y, kind, r_target, mean, sd):
    """Exact means, SDs (ddof=1) and Pearson r by applying ONE small affine map to the
    extras (baby copies + stars) only. The main drawing is never touched, so it is never
    sheared; the map is the closest-to-identity one that satisfies the five equations.
    Returns x, y, residual, singular values of the 2x2 part (1,1 = no distortion)."""
    ex = kind != SHAPE
    ex0, ey0 = x[ex] - 50, y[ex] - 50

    def full(q):
        X, Y = x.copy(), y.copy()
        X[ex] = 50 + q[0] * ex0 + q[1] * ey0 + q[4]
        Y[ex] = 50 + q[2] * ex0 + q[3] * ey0 + q[5]
        return X, Y

    def F(q):
        X, Y = full(q)
        return np.array([X.mean() - mean[0], Y.mean() - mean[1],
                         X.std(ddof=1) - sd[0], Y.std(ddof=1) - sd[1],
                         np.corrcoef(X, Y)[0, 1] - r_target])

    q0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.02, 0.02])      # translations are cheap
    res = minimize(lambda q: np.sum(w * (q - q0) ** 2), q0,
                   constraints=[{"type": "eq", "fun": F}], method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    q = res.x
    for _ in range(30):                                  # Newton polish to machine precision
        f = F(q)
        if np.abs(f).max() < 1e-13:
            break
        J = np.empty((5, 6))
        for j in range(6):
            qq = q.copy()
            qq[j] += 1e-7
            J[:, j] = (F(qq) - f) / 1e-7
        q = q - np.linalg.lstsq(J, f, rcond=None)[0]
    X, Y = full(q)
    return X, Y, float(np.abs(F(q)).max()), np.linalg.svd(q[:4].reshape(2, 2), compute_uv=False)


def stats(x, y):
    r = np.corrcoef(x, y)[0, 1]
    slope, icpt = np.polyfit(x, y, 1)
    return dict(mean_x=x.mean(), mean_y=y.mean(), sd_x=x.std(ddof=1),
                sd_y=y.std(ddof=1), r=r, r2=r * r, slope=slope, intercept=icpt)


# --------------------------------------------------------------------------
PALETTE = ["#e63946", "#d6336c", "#f77f00", "#e09f00", "#2a9d8f", "#fcbf49",
           "#3a86ff", "#7b2cbf", "#6a994e", "#bc4749", "#118ab2", "#2b9348"]


def draw_month(pdf, year, month, x, y, kind, caption, st, colour, out_png, original=False):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#fbf8f1")

    # title
    fig.text(0.06, 0.94, calendar.month_name[month].upper(), fontsize=34,
             fontweight="bold", color=colour, va="center")
    fig.text(0.94, 0.94, str(year), fontsize=34, color="#555", ha="right", va="center")
    sub = (f"“{caption}” — the original, untouched dataset (r = {st['r']:.2f}, R² = {st['r2']:.3f}); "
           "the other months share r = 0.82, R² = 0.67" if original else
           f"“{caption}” — same mean, SD, correlation and R² as every other month except January")
    fig.text(0.06, 0.895, sub,
             fontsize=10.5, color="#555", style="italic")

    # scatter plot
    ax = fig.add_axes([0.06, 0.46, 0.42, 0.42])
    ax.set_facecolor("white")
    px, py = 0.08 * np.ptp(x), 0.08 * np.ptp(y)
    ax.scatter(x[kind == SPARSE], y[kind == SPARSE], s=11, color="#9aa7c7", alpha=0.8, linewidth=0, zorder=2)
    ax.scatter(x[kind == MINI], y[kind == MINI], s=28, color=colour, edgecolor="white", linewidth=0.5, zorder=4)
    ax.scatter(x[kind == SHAPE], y[kind == SHAPE], s=22, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
    glyph, gcol = STAR_GLYPH.get(month, DEFAULT_STAR)
    for u, v in zip(x[kind == STAR], y[kind == STAR]):
        ax.text(u, v, glyph, color=gcol, fontsize=11.5, ha="center", va="center",
                fontfamily="DejaVu Sans", zorder=2)
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
    pn.text(0, 1, "ORIGINAL STATS" if original else "SAME IN EVERY OTHER MONTH", fontsize=9, fontweight="bold", color="#444", va="top")
    for i, (k, v) in enumerate(rows):
        yy = 0.86 - i * 0.105
        pn.text(0, yy, k, fontsize=11, color="#555", va="center")
        pn.text(1, yy, f"{v:.2f}", fontsize=11, color="#222", ha="right", va="center",
                fontfamily="monospace", fontweight="bold" if k == "R²" else "normal")
    pn.text(0, -0.02, f"n = {len(x)} points ({int((kind == SHAPE).sum())} drawing + {int((kind == MINI).sum())} baby + {int((kind == STAR).sum())} {glyph} + {int((kind == SPARSE).sum())} sparse)" if not original else f"n = {len(x)} points (original data, nothing added)", fontsize=9, color="#888", va="center")

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

    mean, sd = (50.0, 50.0), (TARGET_SD, TARGET_SD)

    rows, all_stats = [], []
    with PdfPages(os.path.join(a.out, f"datasaurus_calendar_{a.year}.pdf")) as pdf:
        for m, ((fn, caption), colour) in enumerate(zip(SHAPES, PALETTE), start=1):
            png = os.path.join(a.out, f"{m:02d}_{calendar.month_name[m].lower()}.png")
            if fn is dino:                       # the real Datasaurus: nothing added, nothing moved
                x, y = (np.asarray(v, float) for v in fn())
                kind = np.full(len(x), SHAPE)
                st = stats(x, y)
                print(f"{calendar.month_name[m]:9s} {caption:18s} ORIGINAL data, r={st['r']:.3f}, n={len(x)}")
                draw_month(pdf, a.year, m, x, y, kind, caption, st, colour, png, original=True)
            else:
                x, y, kind = layout(fn, a.r, m, RADIUS.get(caption, SHAPE_RADIUS))
                r_nat = np.corrcoef(x, y)[0, 1]
                x, y, resid, sv = fix_extras(x, y, kind, a.r, mean, sd)
                st = stats(x, y)
                all_stats.append(st)
                print(f"{calendar.month_name[m]:9s} {caption:18s} natural r={r_nat:.3f}  resid={resid:.1e}  "
                      f"extras map sv={sv[0]:.2f},{sv[1]:.2f}  n={len(x)}")
                draw_month(pdf, a.year, m, x, y, kind, caption, st, colour, png)
            rows += [(calendar.month_name[m], f"{u:.6f}", f"{v:.6f}", ("shape", "baby", "star", "sparse")[int(k)])
                     for u, v, k in zip(x, y, kind)]

    with open(os.path.join(a.out, "datasets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "x", "y", "kind"])
        w.writerows(rows)

    keys = all_stats[0].keys()
    spread = {k: max(s[k] for s in all_stats) - min(s[k] for s in all_stats) for k in keys}
    print("max spread across Feb-Dec (should be ~0):")
    for k, v in spread.items():
        print(f"  {k:10s} {all_stats[0][k]: .4f}   spread {v:.2e}")


if __name__ == "__main__":
    main()
