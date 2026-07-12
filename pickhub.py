"""
Pick Hub — MLB engine v1

An Elo rating model for MLB game outcomes, with a starting-pitcher adjustment
based on the pitcher's PRIOR season ERA (prior season, so the backtest has no
lookahead leakage).

Commands
--------
  python pickhub.py fetch 2022 2023 2024 2025    Download games -> games.csv
  python pickhub.py backtest                      Fit + walk-forward test
  python pickhub.py today                         Win probabilities for today
  python pickhub.py selftest                      Verify the math (no network)

Data source: MLB Stats API (statsapi.mlb.com) — official, free, no key.
"""

import sys, json, time, math, os
from collections import defaultdict

import requests

API = "https://statsapi.mlb.com/api/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV = os.path.join(HERE, "games.csv")
ERA_CACHE = os.path.join(HERE, "pitcher_era_cache.json")

# ---------------------------------------------------------------- parameters
# These are the model's dials. `backtest` grid-searches the last two.
HFA_ELO      = 24.0    # home field advantage, in Elo points (~54% home win rate)
K_FACTOR     = 4.0     # Elo update speed. MLB is high-variance, so this is low.
SEASON_CARRY = 0.75    # how much rating carries to next season (0.75 = revert 25% to 1500)
LEAGUE_ERA   = 4.20    # rough MLB average; recomputed from data when possible

# grid-searched in backtest():
ERA_TO_ELO   = 12.0    # Elo points per run of ERA better than league average
ERA_CLAMP    = 60.0    # max Elo swing from one starting pitcher


# ---------------------------------------------------------------- core math
def expected_score(elo_a, elo_b):
    """Probability that A beats B, standard Elo logistic."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def mov_multiplier(run_diff, elo_diff):
    """
    Margin-of-victory multiplier. Blowouts move ratings more than 1-run games,
    but we damp it so a favourite winning big doesn't run away with the rating.
    This is the FiveThirtyEight-style autocorrelation correction.
    """
    rd = abs(run_diff)
    if rd == 0:
        rd = 1
    return math.log(rd + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))


def pitcher_adjustment(prior_era, league_era=LEAGUE_ERA,
                       era_to_elo=ERA_TO_ELO, clamp=ERA_CLAMP):
    """
    Convert a starting pitcher's PRIOR-season ERA into Elo points.
    Better than league average -> positive. Unknown/rookie -> 0 (neutral).
    """
    if prior_era is None:
        return 0.0
    adj = (league_era - prior_era) * era_to_elo
    return max(-clamp, min(clamp, adj))


# ---------------------------------------------------------------- fetch
def fetch_games(seasons):
    """Pull every completed regular-season game for the given seasons."""
    rows = []
    for season in seasons:
        print(f"  fetching {season} schedule...", flush=True)
        url = (f"{API}/schedule?sportId=1&gameType=R"
               f"&startDate={season}-03-01&endDate={season}-11-15"
               f"&hydrate=probablePitcher")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()

        for day in data.get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                h = g["teams"]["home"]
                a = g["teams"]["away"]
                if "score" not in h or "score" not in a:
                    continue
                rows.append({
                    "game_pk":     g["gamePk"],
                    "date":        g["gameDate"][:10],
                    "season":      season,
                    "home":        h["team"]["name"],
                    "away":        a["team"]["name"],
                    "home_score":  h["score"],
                    "away_score":  a["score"],
                    "home_sp_id":  (h.get("probablePitcher") or {}).get("id"),
                    "away_sp_id":  (a.get("probablePitcher") or {}).get("id"),
                    "home_sp":     (h.get("probablePitcher") or {}).get("fullName", ""),
                    "away_sp":     (a.get("probablePitcher") or {}).get("fullName", ""),
                })
        time.sleep(0.4)

    rows.sort(key=lambda x: (x["date"], x["game_pk"]))
    cols = ["game_pk", "date", "season", "home", "away", "home_score", "away_score",
            "home_sp_id", "away_sp_id", "home_sp", "away_sp"]
    with open(GAMES_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r_ in rows:
            f.write(",".join('"%s"' % str(r_[c] if r_[c] is not None else "")
                             for c in cols) + "\n")
    print(f"  wrote {len(rows)} games -> games.csv")
    return rows


def load_games():
    import csv
    if not os.path.exists(GAMES_CSV):
        sys.exit("No games.csv. Run:  python pickhub.py fetch 2022 2023 2024 2025")
    out = []
    with open(GAMES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["season"] = int(row["season"])
            row["home_score"] = int(row["home_score"])
            row["away_score"] = int(row["away_score"])
            for k in ("home_sp_id", "away_sp_id"):
                row[k] = int(row[k]) if row[k] else None
            out.append(row)
    return out


# ---------------------------------------------------------------- pitcher ERA
def load_era_cache():
    if os.path.exists(ERA_CACHE):
        with open(ERA_CACHE) as f:
            return json.load(f)
    return {}


def get_prior_era(pid, season, cache):
    """
    Season-BEFORE-this-one ERA for pitcher `pid`. Using the prior season means
    the backtest never sees information from the future. Returns None if unknown.
    """
    if pid is None:
        return None
    key = f"{pid}:{season - 1}"
    if key in cache:
        return cache[key]

    url = (f"{API}/people/{pid}?hydrate=stats(group=[pitching],"
           f"type=[season],season={season - 1})")
    era = None
    try:
        r = requests.get(url, timeout=30)
        if r.ok:
            people = r.json().get("people", [])
            if people:
                for st in people[0].get("stats", []):
                    for sp in st.get("splits", []):
                        val = sp.get("stat", {}).get("era")
                        if val not in (None, "-", ".---"):
                            era = float(val)
                            break
    except Exception:
        era = None

    cache[key] = era
    with open(ERA_CACHE, "w") as f:
        json.dump(cache, f)
    time.sleep(0.15)
    return era


def hydrate_eras(games):
    """Attach prior-season ERA to every game's two starters. Cached to disk."""
    cache = load_era_cache()
    need = set()
    for g in games:
        for side in ("home", "away"):
            pid = g[f"{side}_sp_id"]
            if pid and f"{pid}:{g['season'] - 1}" not in cache:
                need.add((pid, g["season"]))
    if need:
        print(f"  looking up {len(need)} pitcher seasons (cached after this)...")
        for i, (pid, season) in enumerate(sorted(need), 1):
            get_prior_era(pid, season, cache)
            if i % 50 == 0:
                print(f"    {i}/{len(need)}", flush=True)

    for g in games:
        g["home_sp_era"] = cache.get(f"{g['home_sp_id']}:{g['season'] - 1}")
        g["away_sp_era"] = cache.get(f"{g['away_sp_id']}:{g['season'] - 1}")
    return games


# ---------------------------------------------------------------- the model
def run_elo(games, era_to_elo=ERA_TO_ELO, clamp=ERA_CLAMP,
            hfa=HFA_ELO, k=K_FACTOR, carry=SEASON_CARRY,
            skip_seasons=0, collect=True):
    """
    Walk forward through every game in date order.
    Predict, THEN update ratings. Never the other way round — that's the whole
    point of a walk-forward test.

    Returns (predictions, final_ratings). Predictions from the first
    `skip_seasons` seasons are dropped (burn-in) but still train the ratings.
    """
    elo = defaultdict(lambda: 1500.0)
    league_era = LEAGUE_ERA
    preds = []
    seasons_seen = []
    cur_season = None

    for g in games:
        if g["season"] != cur_season:
            if cur_season is not None:
                for t in elo:
                    elo[t] = 1500.0 + (elo[t] - 1500.0) * carry
            cur_season = g["season"]
            seasons_seen.append(cur_season)

        h, a = g["home"], g["away"]
        h_adj = pitcher_adjustment(g.get("home_sp_era"), league_era, era_to_elo, clamp)
        a_adj = pitcher_adjustment(g.get("away_sp_era"), league_era, era_to_elo, clamp)

        h_eff = elo[h] + hfa + h_adj
        a_eff = elo[a] + a_adj

        p_home = expected_score(h_eff, a_eff)
        home_won = 1 if g["home_score"] > g["away_score"] else 0

        burned_in = len(seasons_seen) > skip_seasons
        if collect and burned_in:
            preds.append({
                "date": g["date"], "home": h, "away": a,
                "p_home": p_home, "home_won": home_won,
                "run_diff": g["home_score"] - g["away_score"],
            })

        # --- update (uses the actual result, applied only after predicting)
        mult = mov_multiplier(g["home_score"] - g["away_score"], h_eff - a_eff)
        shift = k * mult * (home_won - p_home)
        elo[h] += shift
        elo[a] -= shift

    return preds, dict(elo)


# ---------------------------------------------------------------- scoring
def log_loss(preds):
    tot = 0.0
    for p in preds:
        q = min(max(p["p_home"], 1e-9), 1 - 1e-9)
        tot += -(p["home_won"] * math.log(q) + (1 - p["home_won"]) * math.log(1 - q))
    return tot / len(preds)


def brier(preds):
    return sum((p["p_home"] - p["home_won"]) ** 2 for p in preds) / len(preds)


def accuracy(preds):
    hits = sum(1 for p in preds if (p["p_home"] > .5) == (p["home_won"] == 1))
    return hits / len(preds)


def calibration_table(preds, bins=10):
    """The single most important output. Does 60% mean 60%?"""
    buckets = defaultdict(list)
    for p in preds:
        b = min(int(p["p_home"] * bins), bins - 1)
        buckets[b].append(p)
    rows = []
    for b in sorted(buckets):
        grp = buckets[b]
        rows.append({
            "band":      f"{b/bins:.0%}–{(b+1)/bins:.0%}",
            "n":         len(grp),
            "predicted": sum(g["p_home"] for g in grp) / len(grp),
            "actual":    sum(g["home_won"] for g in grp) / len(grp),
        })
    return rows


# ---------------------------------------------------------------- commands
def cmd_backtest():
    games = hydrate_eras(load_games())
    seasons = sorted({g["season"] for g in games})
    print(f"\n{len(games)} games, seasons {seasons[0]}–{seasons[-1]}")
    print("Burning in on the first season; scoring on the rest.\n")

    print("Fitting the pitcher dial (grid search on log loss)...")
    best = None
    for era_to_elo in [0, 4, 8, 12, 16, 20, 24]:
        for clamp in [30, 45, 60, 80]:
            preds, _ = run_elo(games, era_to_elo=era_to_elo, clamp=clamp, skip_seasons=1)
            ll = log_loss(preds)
            if best is None or ll < best[0]:
                best = (ll, era_to_elo, clamp)
    ll, era_to_elo, clamp = best
    print(f"  best: {era_to_elo} Elo pts per run of ERA, clamped at ±{clamp}\n")

    preds, final = run_elo(games, era_to_elo=era_to_elo, clamp=clamp, skip_seasons=1)

    # --- baselines to beat
    base_rate = sum(p["home_won"] for p in preds) / len(preds)
    coin  = [{"p_home": 0.500,     "home_won": p["home_won"]} for p in preds]
    homer = [{"p_home": base_rate, "home_won": p["home_won"]} for p in preds]

    print("=" * 62)
    print("BACKTEST  (walk-forward: every prediction made before the game)")
    print("=" * 62)
    print(f"  Games scored          {len(preds)}")
    print(f"  Home team actually won {base_rate:.1%} of the time\n")
    print(f"  {'':22}{'log loss':>10}{'brier':>9}{'acc':>8}")
    print(f"  {'Coin flip (50/50)':22}{log_loss(coin):>10.4f}{brier(coin):>9.4f}{'—':>8}")
    print(f"  {'Always pick home':22}{log_loss(homer):>10.4f}{brier(homer):>9.4f}{accuracy(homer):>8.1%}")
    print(f"  {'THIS MODEL':22}{ll:>10.4f}{brier(preds):>9.4f}{accuracy(preds):>8.1%}")

    edge = log_loss(homer) - ll
    print(f"\n  Improvement over 'always pick home': {edge:+.4f} log loss")
    if edge <= 0.001:
        print("  >> The model is NOT beating a dumb baseline. Do not ship this.")
    else:
        print("  >> The model beats the dumb baseline. Necessary, not sufficient.")

    print("\n" + "-" * 62)
    print("CALIBRATION — the number that actually matters")
    print("-" * 62)
    print(f"  {'band':>10}{'games':>8}{'we said':>10}{'happened':>11}{'gap':>8}")
    for row in calibration_table(preds):
        gap = row["actual"] - row["predicted"]
        flag = "" if abs(gap) < .04 else "  <-- off"
        print(f"  {row['band']:>10}{row['n']:>8}{row['predicted']:>10.1%}"
              f"{row['actual']:>11.1%}{gap:>+8.1%}{flag}")

    print("\n" + "-" * 62)
    print("TOP 10 TEAMS, final ratings")
    print("-" * 62)
    for t, r in sorted(final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {r:7.0f}  {t}")

    print("\n" + "=" * 62)
    print("READ THIS BEFORE YOU BET A DOLLAR")
    print("=" * 62)
    print("""
  Beating the baselines above is the easy part. The market beats them too,
  by more. What you have NOT tested here is whether this model beats the
  CLOSING LINE — and that is the only test that matters commercially.

  To run that test you need historical odds, which MLB's API does not have.
  Get them from a paid odds archive, join on game_pk + date, and compare
  your probability to the vig-free market probability. If your model has
  no edge over the closing line, it has no edge. Publishing calibrated
  probabilities is still an honest product. Selling them as +EV is not.
""")


def cmd_today():
    games = hydrate_eras(load_games())
    _, elo = run_elo(games, collect=False)
    cache = load_era_cache()

    from datetime import date
    today = date.today().isoformat()
    url = (f"{API}/schedule?sportId=1&gameType=R&date={today}"
           f"&hydrate=probablePitcher")
    data = requests.get(url, timeout=30).json()

    season = date.today().year
    slate = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            h = g["teams"]["home"]["team"]["name"]
            a = g["teams"]["away"]["team"]["name"]
            hp = (g["teams"]["home"].get("probablePitcher") or {})
            ap = (g["teams"]["away"].get("probablePitcher") or {})
            h_era = get_prior_era(hp.get("id"), season, cache)
            a_era = get_prior_era(ap.get("id"), season, cache)

            h_eff = elo.get(h, 1500) + HFA_ELO + pitcher_adjustment(h_era)
            a_eff = elo.get(a, 1500) + pitcher_adjustment(a_era)
            p = expected_score(h_eff, a_eff)
            slate.append((p, h, a, hp.get("fullName", "TBD"),
                          ap.get("fullName", "TBD"), g["gameDate"][11:16]))

    if not slate:
        print(f"No games found for {today}.")
        return

    print(f"\n  MLB — {today}   (model win probability, HOME team)")
    print("  " + "-" * 74)
    for p, h, a, hp, ap, t in sorted(slate, key=lambda x: -abs(x[0] - .5)):
        fair = f"{-100*p/(1-p):.0f}" if p > .5 else f"+{100*(1-p)/p:.0f}"
        print(f"  {t}Z  {a:<24} @ {h:<24}  {p:>6.1%}  fair {fair}")
        print(f"           {ap:<24}   {hp:<24}")
    print("""
  These are model probabilities, not picks. A pick requires an edge over the
  price on the board — compare the fair line above to the actual market line.
  If they agree, there is no bet.
""")


def cmd_selftest():
    """Verify the math without touching the network."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    print("\nSelf-test — verifying the engine's math\n")
    check("equal ratings -> 50%", abs(expected_score(1500, 1500) - 0.5) < 1e-9)
    check("+100 Elo -> ~64%",     abs(expected_score(1600, 1500) - 0.640) < 0.002)
    check("symmetry",             abs(expected_score(1600, 1500) + expected_score(1500, 1600) - 1) < 1e-9)
    check("home edge ~54%",       0.53 < expected_score(1500 + HFA_ELO, 1500) < 0.55)
    check("ace helps",            pitcher_adjustment(2.50) > 0)
    check("bad starter hurts",    pitcher_adjustment(6.00) < 0)
    check("unknown pitcher neutral", pitcher_adjustment(None) == 0.0)
    check("pitcher adj is clamped", pitcher_adjustment(0.01) <= ERA_CLAMP)
    check("blowouts move more",   mov_multiplier(10, 0) > mov_multiplier(1, 0))
    check("favourite blowout damped", mov_multiplier(10, 300) < mov_multiplier(10, 0))

    # Synthetic league: team A is genuinely better; Elo should learn that.
    import random
    random.seed(7)
    fake = []
    for i in range(600):
        hw = random.random() < 0.68          # A beats B 68% of the time
        fake.append({"season": 2024, "date": f"2024-05-{(i%28)+1:02d}",
                     "home": "A", "away": "B",
                     "home_score": 5 if hw else 2, "away_score": 2 if hw else 5,
                     "home_sp_era": None, "away_sp_era": None})
    preds, elo = run_elo(fake, skip_seasons=0)
    learned = expected_score(elo["A"] + HFA_ELO, elo["B"])
    check(f"learns a 68% team (got {learned:.0%})", 0.60 < learned < 0.76)
    check("beats a coin flip on it",
          log_loss(preds[300:]) < log_loss([{"p_home": .5, "home_won": p["home_won"]}
                                            for p in preds[300:]]))

    print(f"\n  {'ALL PASS — the math is sound.' if ok else 'SOMETHING IS BROKEN.'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "fetch":
        yrs = [int(y) for y in sys.argv[2:]] or [2022, 2023, 2024, 2025]
        print(f"Fetching seasons: {yrs}")
        fetch_games(yrs)
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "today":
        cmd_today()
    elif cmd == "selftest":
        sys.exit(cmd_selftest())
    else:
        print(__doc__)
