"""
Pick Hub — the bot.

Runs on GitHub Actions every 15 minutes. Each run:

  1. Refreshes games.csv with any games that finished recently
  2. Rebuilds Elo ratings from the full history
  3. SETTLES any pending picks whose games are now final
  4. PUBLISHES a pick for any game starting in the next ~75 minutes
  5. Writes site/picks.json so the website can show the board

Everything is committed to git. The commit timestamp is the proof that the
pick existed before the game started. That is the entire point.

Nothing is ever edited or deleted from ledger.csv. Rows are appended, and
pending rows are settled in place — never removed.
"""

import csv, json, os, sys, math
from datetime import datetime, timedelta, timezone

import requests

from pickhub import (
    API, expected_score, pitcher_adjustment, run_elo,
    load_era_cache, get_prior_era, HFA_ELO,
)

HERE   = os.path.dirname(os.path.abspath(__file__))
GAMES  = os.path.join(HERE, "games.csv")
LEDGER = os.path.join(HERE, "ledger.csv")
SITE   = os.path.join(HERE, "site", "picks.json")

LEDGER_COLS = [
    "game_pk", "published_utc", "game_start_utc", "sport", "away", "home",
    "away_sp", "home_sp", "side", "p_win", "fair_line",
    "status", "away_score", "home_score", "result",
]

# Publish a pick when the game starts within this window.
# GitHub's scheduler can drift by several minutes, so we use a band, not a point.
LEAD_MIN = 50
LEAD_MAX = 80


# ------------------------------------------------------------------ helpers
def now_utc():
    return datetime.now(timezone.utc)


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ledger(rows):
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLS})


def american(p):
    """Fair moneyline for probability p."""
    if p >= 0.5:
        return f"{round(-100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def schedule(start, end):
    url = (f"{API}/schedule?sportId=1&gameType=R"
           f"&startDate={start}&endDate={end}&hydrate=probablePitcher")
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ 1. refresh
def refresh_games():
    """Append any newly-final games to games.csv. Never rewrites old rows."""
    existing = {}
    if os.path.exists(GAMES):
        with open(GAMES, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["game_pk"]] = row
    else:
        sys.exit("games.csv missing. Run `python pickhub.py fetch 2022 2023 2024 2025` "
                 "once locally and commit the result.")

    today = now_utc().date()
    data = schedule((today - timedelta(days=5)).isoformat(), today.isoformat())

    added = 0
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            pk = str(g["gamePk"])
            if pk in existing:
                continue
            h, a = g["teams"]["home"], g["teams"]["away"]
            if "score" not in h or "score" not in a:
                continue
            existing[pk] = {
                "game_pk": pk,
                "date": g["gameDate"][:10],
                "season": g["gameDate"][:4],
                "home": h["team"]["name"],
                "away": a["team"]["name"],
                "home_score": h["score"],
                "away_score": a["score"],
                "home_sp_id": (h.get("probablePitcher") or {}).get("id", ""),
                "away_sp_id": (a.get("probablePitcher") or {}).get("id", ""),
                "home_sp": (h.get("probablePitcher") or {}).get("fullName", ""),
                "away_sp": (a.get("probablePitcher") or {}).get("fullName", ""),
            }
            added += 1

    if added:
        cols = ["game_pk", "date", "season", "home", "away", "home_score",
                "away_score", "home_sp_id", "away_sp_id", "home_sp", "away_sp"]
        rows = sorted(existing.values(), key=lambda r: (r["date"], int(r["game_pk"])))
        with open(GAMES, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    print(f"[refresh] {added} new final games")


# ------------------------------------------------------------------ 2. ratings
def build_ratings():
    from pickhub import load_games, hydrate_eras
    games = hydrate_eras(load_games())
    _, elo = run_elo(games, collect=False)
    print(f"[ratings] built from {len(games)} games")
    return elo


# ------------------------------------------------------------------ 3. settle
def settle(ledger):
    pending = [r for r in ledger if r["status"] == "PENDING"]
    if not pending:
        print("[settle] nothing pending")
        return 0

    today = now_utc().date()
    data = schedule((today - timedelta(days=4)).isoformat(),
                    (today + timedelta(days=1)).isoformat())
    finals = {}
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                h, a = g["teams"]["home"], g["teams"]["away"]
                if "score" in h and "score" in a:
                    finals[str(g["gamePk"])] = (a["score"], h["score"])

    n = 0
    for r in pending:
        if r["game_pk"] not in finals:
            continue
        away_s, home_s = finals[r["game_pk"]]
        winner = r["home"] if home_s > away_s else r["away"]
        r["away_score"] = away_s
        r["home_score"] = home_s
        r["status"] = "SETTLED"
        r["result"] = "WIN" if winner == r["side"] else "LOSS"
        n += 1
        print(f"[settle] {r['away']} @ {r['home']}  {away_s}-{home_s}  -> {r['result']}")
    print(f"[settle] settled {n}")
    return n


# ------------------------------------------------------------------ 4. publish
def publish(ledger, elo):
    seen = {r["game_pk"] for r in ledger}
    cache = load_era_cache()
    t = now_utc()
    season = t.year

    data = schedule(t.date().isoformat(), (t + timedelta(days=1)).date().isoformat())

    added = 0
    for day in data.get("dates", []):
        for g in day.get("games", []):
            pk = str(g["gamePk"])
            if pk in seen:
                continue
            if g.get("status", {}).get("abstractGameState") != "Preview":
                continue

            start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
            mins = (start - t).total_seconds() / 60
            if not (LEAD_MIN <= mins <= LEAD_MAX):
                continue

            ht = g["teams"]["home"]["team"]["name"]
            at = g["teams"]["away"]["team"]["name"]
            hp = g["teams"]["home"].get("probablePitcher") or {}
            ap = g["teams"]["away"].get("probablePitcher") or {}

            h_era = get_prior_era(hp.get("id"), season, cache)
            a_era = get_prior_era(ap.get("id"), season, cache)

            h_eff = elo.get(ht, 1500.0) + HFA_ELO + pitcher_adjustment(h_era)
            a_eff = elo.get(at, 1500.0) + pitcher_adjustment(a_era)
            p_home = expected_score(h_eff, a_eff)

            side  = ht if p_home >= 0.5 else at
            p_win = p_home if p_home >= 0.5 else 1 - p_home

            ledger.append({
                "game_pk": pk,
                "published_utc": t.replace(microsecond=0).isoformat(),
                "game_start_utc": start.replace(microsecond=0).isoformat(),
                "sport": "MLB",
                "away": at, "home": ht,
                "away_sp": ap.get("fullName", "TBD"),
                "home_sp": hp.get("fullName", "TBD"),
                "side": side,
                "p_win": f"{p_win:.4f}",
                "fair_line": american(p_win),
                "status": "PENDING",
                "away_score": "", "home_score": "", "result": "",
            })
            added += 1
            print(f"[publish] {side} ({p_win:.1%}) — {at} @ {ht}, starts in {mins:.0f} min")

    print(f"[publish] added {added}")
    return added


# ------------------------------------------------------------------ 5. site
def write_site(ledger):
    settled = [r for r in ledger if r["status"] == "SETTLED"]
    wins   = sum(1 for r in settled if r["result"] == "WIN")
    losses = len(settled) - wins

    recent = sorted(ledger, key=lambda r: r["published_utc"], reverse=True)[:40]

    out = {
        "updated_utc": now_utc().replace(microsecond=0).isoformat(),
        "record":   f"{wins}-{losses}" if settled else "0-0",
        "settled":  len(settled),
        "hit_rate": round(100 * wins / len(settled), 1) if settled else None,
        "picks": [
            {
                "sport":     r["sport"],
                "matchup":   f"{r['away']} @ {r['home']}",
                "side":      r["side"],
                "confidence": round(float(r["p_win"]) * 100, 1),
                "fair_line": r["fair_line"],
                "published": r["published_utc"],
                "starts":    r["game_start_utc"],
                "status":    r["status"],
                "result":    r["result"],
                "score":     (f"{r['away_score']}-{r['home_score']}"
                              if r["status"] == "SETTLED" else ""),
            }
            for r in recent
        ],
    }
    os.makedirs(os.path.dirname(SITE), exist_ok=True)
    with open(SITE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[site] record {out['record']}, {len(recent)} picks on the board")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    refresh_games()
    elo = build_ratings()
    ledger = read_ledger()

    settle(ledger)
    publish(ledger, elo)

    ledger.sort(key=lambda r: r["published_utc"])
    write_ledger(ledger)
    write_site(ledger)
    print("[done]")
