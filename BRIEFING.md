# Team-news briefing: instructions for Claude

You are writing `team_news.json`, a short team-news briefing for an FPL dashboard. The app
uses each player's start chance in this gameweek's projections, so be accurate and cautious.

## Inputs
1. `news/latest.json` on the `news-data` branch: headlines and snippets per player and club,
   collected by the "Collect team news" workflow. Fetch it with:
   `git fetch --depth=1 origin news-data && git show origin/news-data:news/latest.json > /tmp/news.json`
2. Current FPL data (status, chance of playing, news) for context: `python3 -c "import server; ..."`
   or the `fpl_status`, `fpl_chance` and `fpl_news` fields already in the news file.

## When to stop without writing anything
- The news file's `gameweek` is not the next FPL gameweek, or `collected_at` is more than 6 hours old.
- The deadline is more than 36 hours away, or has passed.
- `team_news.json` already exists for this gameweek and nothing material has changed since.

## What to write
Only players where the news says something about **whether they will start or play** this gameweek:
injuries, fitness, suspensions, returns to training, rotation, a rival back fit, manager quotes on
selection. Skip transfer gossip, opinion pieces without new information, and match reports.

```json
{
  "gameweek": 11,
  "generated_at": "2026-11-20T15:04:00Z",
  "news_collected_at": "<collected_at from the news file>",
  "summary": "2-4 sentences: the most important team news for these squads.",
  "players": [
    {
      "id": 123,
      "name": "Konsa",
      "start_chance": 35,
      "confidence": "high",
      "verdict": "Likely benched: Arteta said Saliba is fit and 'ready to start'.",
      "sources": [{"title": "…", "source": "BBC Sport", "link": "https://…"}]
    }
  ]
}
```

- `id`: the FPL player id (the keys of `players` in the news file). Never invent ids.
- `start_chance`: 0-100, your estimate that they start this gameweek. Use `null` if the news
  matters but you can't estimate it.
- `confidence`:
  - `high`: the manager or the club said it directly (press conference, official site), or two
    or more major outlets report it.
  - `medium`: one reputable outlet (BBC, Guardian, Sky, The Athletic, major newspapers).
  - `low`: fan sites, rumours, or inference from a single unclear headline.
- `verdict`: one plain sentence a fan would understand, naming the evidence.
- `sources`: up to 3, best first, copied from the news file. Never make up a link.
- Weigh fan sites below official and national sources, and never use them as the only basis for
  `high` confidence.
- Don't repeat what FPL's own injury flag already says unless the news changes it.

## Finish
1. Run `python3 validate_briefing.py team_news.json`. Fix any errors it reports.
2. Commit only `team_news.json` to `main` with the message `Team-news briefing for GW<n>` and push.
   The push rebuilds the dashboard.
3. Reply with the summary and the list of verdicts.
