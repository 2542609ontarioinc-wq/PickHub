# Pick Hub — MLB engine v1

An Elo model for MLB game outcomes. Free data, no API key, runs on your laptop.

---

## What it does

- Pulls every completed MLB game from MLB's official free API
- Rates every team with Elo, updated game by game
- Adjusts each game for the two starting pitchers, using their **prior season** ERA
- **Walk-forward backtests** it: every prediction is made *before* the game, using
  only information that existed at the time. No lookahead. No leakage.
- Reports whether it's **calibrated** — when it says 60%, does that happen 60% of the time?
- Prints today's slate with a fair moneyline for each game

---

## Setup (about 5 minutes)

**1. Install Python** — get it from python.org. On the installer, tick
   *"Add Python to PATH"*.

**2. Open a terminal** in this folder.
   - Windows: type `cmd` in the folder's address bar, hit Enter
   - Mac: right-click the folder → Services → New Terminal at Folder

**3. Install the one dependency:**
```
pip install requests
```

**4. Check the math is intact** (takes one second, no internet needed):
```
python pickhub.py selftest
```
You should see `ALL PASS`.

---

## Use it

**Download four seasons of games** (takes ~1 minute):
```
python pickhub.py fetch 2022 2023 2024 2025
```

**Backtest** (first run is slow — it looks up ~600 pitchers, then caches them forever):
```
python pickhub.py backtest
```

**Today's games:**
```
python pickhub.py today
```

---

## How to read the backtest

Three numbers come out. Only one of them matters much.

**Log loss / Brier score** — lower is better. The model is compared against two
dumb baselines: a coin flip, and "always pick the home team." If it can't beat
*those*, it's broken. Beating them is the bare minimum, not an achievement — the
betting market beats them by a wide margin too.

**Calibration table** — this is the real test. It buckets every prediction by
confidence and shows what actually happened:

```
      band   games   we said   happened      gap
    50%–60%    1204     54.8%      55.1%    +0.3%
    60%–70%     641     64.2%      61.9%    -2.3%
```

If "we said 64%" and "it happened 62%," you're roughly calibrated. If you said 64%
and it happened 51%, your confidence numbers are fiction and you must not put them
in front of a paying customer.

**Team ratings** — a sanity check. If a 100-loss team is top of your list, something
is wrong with the code, not with baseball.

---

## The thing this does NOT tell you

**It does not tell you whether the model beats the market.**

That's a different, harder test, and it's the only one that matters commercially.
It needs historical closing odds, which MLB does not publish. You'd have to buy them
from an odds archive, join them on `game_pk`, strip the vig out of the market price,
and compare.

Expect the market to win. MLB moneylines are among the sharpest markets in sport,
priced by people with better data than this. A calibrated model that *doesn't* beat
the close is a normal, respectable outcome — and it's still an honest product, as
long as you sell it as **"here are calibrated probabilities"** and not as
**"here is free money."**

The moment you claim an edge you haven't measured, you're Pick1.

---

## What's missing from v1 (in the order I'd add it)

1. **Bullpen strength.** Starters throw ~5 innings now. Ignoring relievers is the
   biggest hole in this model.
2. **Park factors.** Coors Field is not Oracle Park.
3. **Better pitcher inputs.** Prior-season ERA is crude and noisy. FIP, xFIP or
   SIERA over a rolling window would be sharper — but watch for leakage.
4. **Rest and travel.** Bullpen usage over the last three days; getaway-day games.
5. **Market comparison.** Wire in a live odds feed, then log closing line value on
   every pick. This is what turns the model into a product.

---

## Files

| File | What it is |
|---|---|
| `pickhub.py` | The whole engine |
| `games.csv` | Created by `fetch` — the raw game log |
| `pitcher_era_cache.json` | Created automatically — so you only look up each pitcher once |

Data: MLB Stats API (`statsapi.mlb.com`) — official, free, no key.
Not affiliated with or endorsed by MLB.
