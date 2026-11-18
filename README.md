# FPL Coach

A personal Fantasy Premier League assistant that runs on your own computer. It reads the public FPL API, so you don't need to log in.

## Hosted on GitHub (no computer needed)

This repo builds and publishes the dashboard with GitHub Actions and GitHub Pages.

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. List the team IDs to follow in **`teams.json`**, e.g. `[633619, 258412]`. In the dashboard, **⚙ Settings → Edit teams on GitHub** opens this file. Switch between teams with the toggle at the top.
3. **Actions → Update FPL Coach → Run workflow.** When it finishes, your dashboard is at `https://<your-username>.github.io/<repo-name>/`.

It refreshes automatically every morning, on Friday evening after the press conferences, and on Saturday morning. To refresh now, use the **Refresh signals + advice** button on the Signals tab, which opens the *Run workflow* page.

## Run it locally instead

You need Python 3.9 or newer. No other installs are required.

```bash
cd fpl-coach
python3 server.py
```

Open http://127.0.0.1:8765 and enter your team ID. That's the number in your FPL web address: `fantasy.premierleague.com/entry/1234567/event/12`.

## What it does

| Tab | What you get |
|---|---|
| This week | A game plan for the gameweek: transfers, captain, lineup changes, chips and injury checks |
| My team | Best XI and formation, bench order, captain and vice, and your squad's next 6 fixtures |
| Transfers | Roll or transfer; best single and double moves (including −4 hits); sell watchlist; who to buy (affordable, best overall, differentials) |
| Signals | **Refresh signals** button: availability changes since your last check, minutes trends, FPL injury news, headlines mentioning your players, most bought and sold |
| Chips | When to play Wildcard, Free Hit, Bench Boost and Triple Captain, with week-by-week value and window deadlines |
| Fixtures | Difficulty ticker for all 20 teams, sortable by attack or clean-sheet potential, with blanks and doubles shown |
| Players | Searchable projections for about 260 players |
| History | Your points against the average, bench points wasted, hits taken, rank |

### Weekly routine
1. **After the gameweek ends:** open the app and check *This week* and *Transfers*.
2. **Friday, after the press conferences:** go to *Signals* and press **Refresh signals + advice**. The FPL injury flags update after the pressers, and the projections are re-run with the new news.
3. **Before the deadline:** refresh once more and set your captain.

## How the projections work
Expected points (xPts) for each player and fixture are built from:
- **Season stats:** xG, xA, bonus, saves and defensive contributions, each per 90 minutes, steadied for small samples
- **Minutes:** start rate and minutes per start, multiplied by the chance of playing (injured players are assumed to recover over about 3 weeks)
- **Opponent:** team attack and defence strength, blending real goals this season with FPL's strength ratings, plus home advantage
- **Form:** a small adjustment, then a light calibration against FPL's own next-week estimate

Then:
- **Lineup:** every legal formation is tried and the best is picked.
- **Transfers:** your best XI plus captain is simulated over 5 gameweeks, with later weeks weighted a little less, and moves must fit your budget and the 3-per-club rule.
- **Chips:** each gameweek in the chip's window is compared, and Wildcard and Free Hit squads are built with a budget-aware optimiser.

These are estimates. Use them as a strong second opinion, not gospel.

## Limitations
- Free transfers are estimated from your history.
- Transfers you've already made for next week only show after the deadline, because the public API doesn't reveal them before then.
- Selling prices are estimated from your transfer history.
