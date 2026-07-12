"""
Pick Hub — calibration.

The raw Elo model is overconfident: it says 64%, reality is 60%. This is normal
and it is fixable. We fit a logistic recalibration (Platt scaling):

    p_calibrated = sigmoid( a * logit(p_raw) + b )

`a` < 1 shrinks probabilities toward 50%. `b` corrects any home/away bias.

Crucially, a and b are fitted on EARLY seasons and tested on the LAST season,
which the fit never saw. Fitting and testing on the same data would produce a
beautiful calibration table and a worthless model.

  python calibrate.py fit    -> fits, tests out-of-sample, writes calibration.json
  python calibrate.py check  -> prints the before/after tables
"""

import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_FILE = os.path.join(HERE, "calibration.json")

EPS = 1e-6


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def apply_calibration(p, a, b):
    return sigmoid(a * logit(p) + b)


def load_calibration():
    """Returns (a, b). Identity (1, 0) if not yet fitted."""
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE) as f:
            d = json.load(f)
        return d["a"], d["b"]
    return 1.0, 0.0


# ------------------------------------------------------------------ fitting
def log_loss_ab(preds, a, b):
    tot = 0.0
    for p in preds:
        q = min(max(apply_calibration(p["p_home"], a, b), EPS), 1 - EPS)
        y = p["home_won"]
        tot += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return tot / len(preds)


def fit(preds, iters=300, lr=0.5):
    """
    Gradient descent on log loss. Two parameters, convex problem — this
    converges reliably and needs no external library.
    """
    a, b = 1.0, 0.0
    n = len(preds)
    for _ in range(iters):
        ga = gb = 0.0
        for p in preds:
            x = logit(p["p_home"])
            q = sigmoid(a * x + b)
            err = q - p["home_won"]        # gradient of log loss wrt (a*x+b)
            ga += err * x
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


# ------------------------------------------------------------------ reporting
def calib_table(preds, a=1.0, b=0.0, bins=10):
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in preds:
        q = apply_calibration(p["p_home"], a, b)
        buckets[min(int(q * bins), bins - 1)].append((q, p["home_won"]))
    rows = []
    for k in sorted(buckets):
        grp = buckets[k]
        rows.append((
            f"{k/bins:.0%}–{(k+1)/bins:.0%}",
            len(grp),
            sum(g[0] for g in grp) / len(grp),
            sum(g[1] for g in grp) / len(grp),
        ))
    return rows


def print_table(title, rows):
    print(f"\n  {title}")
    print(f"  {'band':>10}{'games':>8}{'we said':>10}{'happened':>11}{'gap':>8}")
    for band, n, said, happened in rows:
        gap = happened - said
        flag = "" if abs(gap) < .03 or n < 30 else "  <-- off"
        print(f"  {band:>10}{n:>8}{said:>10.1%}{happened:>11.1%}{gap:>+8.1%}{flag}")


def weighted_miscal(rows, min_n=30):
    """Average absolute gap, weighted by games. Lower is better."""
    rows = [r for r in rows if r[1] >= min_n]
    tot = sum(r[1] for r in rows)
    return sum(r[1] * abs(r[3] - r[2]) for r in rows) / tot if tot else 0.0


# ------------------------------------------------------------------ commands
def get_preds():
    from pickhub import load_games, hydrate_eras, run_elo
    games = hydrate_eras(load_games())
    preds, _ = run_elo(games, era_to_elo=8.0, clamp=30.0, skip_seasons=1)
    seasons = sorted({g["season"] for g in games})
    # tag each prediction with its season so we can split by time
    for p in preds:
        p["season"] = int(p["date"][:4])
    return preds, seasons


def cmd_fit():
    preds, seasons = get_preds()
    last = seasons[-1]
    train = [p for p in preds if p["season"] < last]
    test  = [p for p in preds if p["season"] == last]

    if len(test) < 200:
        sys.exit("Not enough games in the final season to test on. Fetch more seasons.")

    print(f"\n  Fitting on {seasons[1]}–{last-1}  ({len(train)} games)")
    print(f"  Testing on {last}, which the fit never sees  ({len(test)} games)\n")

    a, b = fit(train)

    before = log_loss_ab(test, 1.0, 0.0)
    after  = log_loss_ab(test, a, b)

    rows_before = calib_table(test)
    rows_after  = calib_table(test, a, b)

    print("=" * 62)
    print("OUT-OF-SAMPLE RESULT")
    print("=" * 62)
    print(f"  shrinkage a = {a:.3f}   (below 1.0 means it pulls toward 50%)")
    print(f"  bias      b = {b:+.3f}")
    print(f"\n  log loss before  {before:.4f}")
    print(f"  log loss after   {after:.4f}   ({after - before:+.4f})")
    print(f"\n  miscalibration before  {weighted_miscal(rows_before):.2%}")
    print(f"  miscalibration after   {weighted_miscal(rows_after):.2%}")

    print_table("BEFORE — raw model", rows_before)
    print_table("AFTER  — calibrated", rows_after)

    if after < before:
        with open(CAL_FILE, "w") as f:
            json.dump({
                "a": a, "b": b,
                "fitted_on": f"{seasons[1]}-{last-1}",
                "tested_on": last,
                "log_loss_before": round(before, 4),
                "log_loss_after": round(after, 4),
            }, f, indent=2)
        print(f"\n  Improved out-of-sample. Written to calibration.json.")
        print(f"  The bot will apply it to every pick from now on.\n")
    else:
        print("\n  Did NOT improve out-of-sample. Not saving.")
        print("  The raw model was already as calibrated as this can make it.\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        cmd_fit()
    elif cmd == "check":
        preds, seasons = get_preds()
        a, b = load_calibration()
        print(f"\n  current calibration: a={a:.3f}, b={b:+.3f}")
        print_table("with calibration applied", calib_table(preds, a, b))
        print()
    else:
        print(__doc__)
