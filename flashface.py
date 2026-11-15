#!/usr/bin/env python3
"""FlashFace: a tiny spaced-repetition app for learning people's names and faces.

Usage:  python3 flashface.py [--port 8000] [--cards cards.csv] [--reviews reviews.csv]
Then open http://localhost:8000/ in a browser.

cards.csv (id,name,image_url) is maintained by hand and reloaded when it changes.
An optional info column holds extra facts about the person, which are shown
below the name on the answer side but are not meant to be memorized.
reviews.csv (timestamp,card_id,grade,reveal_ms) is an append-only log of reviews.
The scheduling state of each card is never stored: it is rebuilt by replaying the
review log, so changing the algorithm below re-interprets the whole history.

Written by Claude (Opus 5.5, Anthropic), November 2026.
"""

import argparse
import csv
import dataclasses
import hashlib
import html
import os
import random
import sys
import threading
from datetime import date, datetime, time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

# --- Scheduler parameters ----------------------------------------------------

LEARNING_STEPS = [timedelta(minutes=1), timedelta(minutes=10)]
RELEARNING_STEPS = [timedelta(minutes=10)]
GRADUATING_INTERVAL = 1  # days, after Good on the last learning step
EASY_INTERVAL = 4  # days, after Easy on a learning card
STARTING_EASE = 2.5
MIN_EASE = 1.3
HARD_FACTOR = 1.2
EASY_BONUS = 1.3
LAPSE_FACTOR = 0.5  # interval multiplier when a review card is forgotten
FUZZ = 0.05  # +/- fraction of random spread added to review intervals
NEW_PER_DAY = 20
LEARN_AHEAD = timedelta(minutes=20)
DAY_START_HOUR = 4  # local time at which a new day starts

GRADES = ("again", "hard", "good", "easy")
REVIEWS_HEADER = ["timestamp", "card_id", "grade", "reveal_ms"]


# --- Scheduler ---------------------------------------------------------------


@dataclasses.dataclass
class CardState:
    phase: str = "new"  # new | learning | review | relearning
    step: int = 0  # index into the (re)learning steps
    interval: int = 0  # days
    ease: float = STARTING_EASE
    due: datetime | None = None
    reps: int = 0
    introduced: datetime | None = None  # time of the first review


def now_local() -> datetime:
    return datetime.now().astimezone()


def day_of(t: datetime) -> date:
    """The scheduling day a moment belongs to (days start at DAY_START_HOUR)."""
    return (t.astimezone() - timedelta(hours=DAY_START_HOUR)).date()


def day_start(d: date) -> datetime:
    return datetime.combine(d, time(DAY_START_HOUR)).astimezone()


def fuzz(days: float, card_id: str, rep: int) -> float:
    """Spread intervals a little, deterministically so that replays are stable."""
    h = hashlib.sha256(f"{card_id}:{rep}".encode()).digest()
    r = int.from_bytes(h[:8]) / 2**64
    return days * (1 + FUZZ * (2 * r - 1))


def review_intervals(s: CardState, card_id: str) -> dict[str, int]:
    """Next interval in days for hard/good/easy on a card in the review phase."""
    rep = s.reps + 1
    hard = max(s.interval + 1, round(fuzz(s.interval * HARD_FACTOR, card_id, rep)))
    good = max(hard + 1, round(fuzz(s.interval * s.ease, card_id, rep)))
    easy = max(good + 1, round(fuzz(s.interval * s.ease * EASY_BONUS, card_id, rep)))
    return {"hard": hard, "good": good, "easy": easy}


def schedule(s: CardState, grade: str, now: datetime, card_id: str) -> CardState:
    """Return the new state of a card after it was reviewed with `grade` at `now`."""
    intervals = review_intervals(s, card_id) if s.phase == "review" else {}
    s = dataclasses.replace(s, reps=s.reps + 1)
    if s.introduced is None:
        s.introduced = now

    def graduate(interval: int) -> None:
        s.phase, s.step, s.interval = "review", 0, interval
        s.due = day_start(day_of(now) + timedelta(days=interval))

    if s.phase == "review":
        if grade == "again":
            s.ease = max(MIN_EASE, s.ease - 0.20)
            s.interval = max(1, round(s.interval * LAPSE_FACTOR))
            s.phase, s.step = "relearning", 0
            s.due = now + RELEARNING_STEPS[0]
        else:
            interval = intervals[grade]
            if grade == "hard":
                s.ease = max(MIN_EASE, s.ease - 0.15)
            elif grade == "easy":
                s.ease += 0.15
            graduate(interval)
        return s

    # new, learning or relearning
    relearning = s.phase == "relearning"
    steps = RELEARNING_STEPS if relearning else LEARNING_STEPS
    if s.phase == "new":
        s.phase, s.step = "learning", 0
    if grade == "again":
        s.step = 0
    elif grade == "good":
        s.step += 1
    if grade == "easy":
        graduate(s.interval + 1 if relearning else EASY_INTERVAL)
    elif s.step >= len(steps):
        graduate(s.interval if relearning else GRADUATING_INTERVAL)
    else:
        s.due = now + steps[s.step]
    return s


def format_delay(delta: timedelta) -> str:
    minutes = delta.total_seconds() / 60
    if minutes < 60:
        return f"{max(1, round(minutes))}m"
    if minutes < 60 * 24:
        return f"{round(minutes / 60)}h"
    days = minutes / (60 * 24)
    if days < 30:
        return f"{round(days)}d"
    if days < 365:
        return f"{days / 30:.1f}mo"
    return f"{days / 365:.1f}y"


def preview(s: CardState, now: datetime, card_id: str) -> dict[str, str]:
    """How long until the card comes back, for each grade (shown on the buttons)."""
    out = {}
    for grade in GRADES:
        new = schedule(s, grade, now, card_id)
        out[grade] = format_delay(timedelta(days=new.interval) if new.phase == "review" else new.due - now)
    return out


# --- Data files and card selection --------------------------------------------


class App:
    def __init__(self, cards_path: Path, reviews_path: Path):
        self.cards_path = cards_path
        self.reviews_path = reviews_path
        self.lock = threading.Lock()
        self.cards: dict[str, dict[str, str]] = {}
        self.cards_mtime: float | None = None
        self.reviews: list[tuple[datetime, str, str]] = []
        self.states: dict[str, CardState] = {}
        self.last_card: str | None = None
        self.load_reviews()
        self.refresh_cards()

    def refresh_cards(self) -> None:
        """(Re)load cards.csv if it changed since the last load."""
        try:
            mtime = self.cards_path.stat().st_mtime
            if mtime == self.cards_mtime:
                return
            cards = {}
            with open(self.cards_path, newline="", encoding="utf-8-sig") as f:
                for line, row in enumerate(csv.DictReader(f), start=2):
                    cid = (row.get("id") or "").strip()
                    if not cid:
                        continue
                    if cid in cards:
                        print(f"warning: {self.cards_path}:{line}: duplicate id {cid!r} ignored", file=sys.stderr)
                        continue
                    cards[cid] = {key: (row.get(key) or "").strip() for key in ("name", "info", "image_url")}
        except (OSError, csv.Error) as e:
            if self.cards_mtime is None:
                raise SystemExit(f"error: cannot read {self.cards_path}: {e}")
            print(f"warning: cannot reload {self.cards_path}, keeping old cards: {e}", file=sys.stderr)
            return
        self.cards, self.cards_mtime = cards, mtime
        print(f"Loaded {len(cards)} cards from {self.cards_path}", file=sys.stderr)
        self.rebuild()

    def load_reviews(self) -> None:
        if not self.reviews_path.exists() or self.reviews_path.stat().st_size == 0:
            with open(self.reviews_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(REVIEWS_HEADER)
            return
        with open(self.reviews_path, newline="", encoding="utf-8") as f:
            for line, row in enumerate(csv.DictReader(f), start=2):
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.astimezone()
                    if row["grade"] not in GRADES:
                        raise ValueError(f"unknown grade {row['grade']!r}")
                    self.reviews.append((ts, row["card_id"], row["grade"]))
                except (KeyError, TypeError, ValueError) as e:
                    print(f"warning: {self.reviews_path}:{line}: skipped: {e}", file=sys.stderr)
        self.reviews.sort(key=lambda r: r[0])
        print(f"Loaded {len(self.reviews)} reviews from {self.reviews_path}", file=sys.stderr)

    def rebuild(self) -> None:
        """Replay the review log to compute the state of every card."""
        self.states = {cid: CardState() for cid in self.cards}
        for ts, cid, grade in self.reviews:
            if cid in self.states:
                self.states[cid] = schedule(self.states[cid], grade, ts, cid)

    def record(self, card_id: str, grade: str, reveal_ms: str, now: datetime) -> None:
        with open(self.reviews_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([now.isoformat(timespec="seconds"), card_id, grade, reveal_ms])
            f.flush()
            os.fsync(f.fileno())
        self.reviews.append((now, card_id, grade))
        self.states[card_id] = schedule(self.states[card_id], grade, now, card_id)
        self.last_card = card_id

    def queue(self, now: datetime) -> tuple[str | None, dict[str, int], datetime | None]:
        """Pick the next card to show; also return counts and when the next card is due."""
        today = day_of(now)
        learning, review, new = [], [], []
        introduced_today = 0
        for cid, s in self.states.items():
            if s.phase == "new":
                new.append(cid)
            elif s.phase == "review":
                review.append((s.due, cid))
            else:
                learning.append((s.due, cid))
            if s.introduced and day_of(s.introduced) == today:
                introduced_today += 1
        learning.sort()
        review.sort()
        random.shuffle(new)
        new_left = min(len(new), max(0, NEW_PER_DAY - introduced_today))

        due_learning = [cid for due, cid in learning if due <= now]
        due_review = [cid for due, cid in review if due <= now]
        ahead = [cid for due, cid in learning if now < due <= now + LEARN_AHEAD]
        candidates = due_learning + due_review + new[:new_left] + ahead
        # Avoid showing the card that was just graded, unless it is the only one.
        others = [cid for cid in candidates if cid != self.last_card]
        card = (others or candidates or [None])[0]

        upcoming = [due for due, _ in learning + review if due > now]
        if len(new) > new_left:
            upcoming.append(day_start(today + timedelta(days=1)))
        stats = {"learning": len(learning), "due": len(due_review), "new": new_left}
        return card, stats, min(upcoming, default=None)


# --- HTML ----------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; --bg: #f5f4f0; --fg: #1d1d1f; --muted: #6e6e73;
        --card: #fff; --border: #d8d8dc; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161618; --fg: #ececee; --muted: #9a9aa0; --card: #242427; --border: #3a3a3e; }
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
       background: var(--bg); color: var(--fg); font: 16px/1.4 system-ui, sans-serif; }
main { width: min(100%, 34rem); padding: 1.5rem 1rem; text-align: center; }
.stats { margin: 0 0 1rem; color: var(--muted); font-size: .9rem; }
.stats b { color: var(--fg); font-weight: 600; }
#face { display: block; margin: 0 auto; max-width: 100%; max-height: 60vh; border-radius: 12px;
        background: var(--card); box-shadow: 0 4px 20px rgb(0 0 0 / .15); }
.panel { min-height: 8.5rem; padding-top: 1.25rem; }
.panel.with-info { min-height: 10.5rem; }
h1 { margin: 0 0 1rem; font-size: 1.8rem; }
h1:has(+ .info) { margin-bottom: .25rem; }
.info { margin: 0 0 1rem; font-size: 1.2rem; color: var(--muted); }
button { font: inherit; color: var(--fg); background: var(--card); border: 1px solid var(--border);
         border-radius: 8px; padding: .6rem 1.2rem; cursor: pointer; }
button:hover { filter: brightness(.95); }
form { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; }
form button { display: flex; flex-direction: column; align-items: center; gap: .1rem;
              padding: .5rem .25rem; border-top: 4px solid var(--c); }
.again { --c: #d9534f; } .hard { --c: #e8a33d; } .good { --c: #4caf50; } .easy { --c: #3b82f6; }
small, kbd { color: var(--muted); font-size: .75rem; }
kbd { font-family: inherit; border: 1px solid var(--border); border-radius: 4px; padding: 0 .3rem; }
"""

SCRIPT = """
const face = document.getElementById("face");
const show = document.getElementById("show");
const answer = document.getElementById("answer");
const form = answer.querySelector("form");
let start = performance.now(), submitted = false;
if (!face.complete) face.addEventListener("load", () => { start = performance.now(); });

function reveal() {
  document.getElementById("reveal_ms").value = Math.round(performance.now() - start);
  show.hidden = true;
  answer.hidden = false;
}
function grade(value) {
  form.requestSubmit(form.querySelector(`button[value="${value}"]`));
}
show.addEventListener("click", reveal);
form.addEventListener("submit", (e) => { if (submitted) e.preventDefault(); submitted = true; });
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || submitted) return;
  if (answer.hidden) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); reveal(); }
  } else if (e.key >= "1" && e.key <= "4") {
    e.preventDefault(); grade(["again", "hard", "good", "easy"][e.key - 1]);
  } else if ((e.key === "Enter" || e.key === " ") && !form.contains(e.target)) {
    e.preventDefault(); grade("good");
  }
});
"""


def page(body: str, head: str = "") -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlashFace</title>
{head}<style>{CSS}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
""".encode()


def stats_html(stats: dict[str, int]) -> str:
    return (f'<p class="stats"><b>{stats["learning"]}</b> learning · '
            f'<b>{stats["due"]}</b> due · <b>{stats["new"]}</b> new</p>')


def card_page(card_id: str, card: dict[str, str], stats: dict[str, int], intervals: dict[str, str]) -> bytes:
    e = html.escape
    buttons = "\n".join(
        f'<button name="grade" value="{g}" class="{g}"><span><kbd>{i}</kbd> {g.capitalize()}</span>'
        f"<small>{intervals[g]}</small></button>"
        for i, g in enumerate(GRADES, start=1)
    )
    info = card.get("info", "")
    info_html = f'\n<p class="info">{e(info)}</p>' if info else ""
    return page(f"""{stats_html(stats)}
<img id="face" src="{e(card["image_url"])}" alt="Who is this?" referrerpolicy="no-referrer">
<div class="panel{" with-info" if info else ""}">
<button id="show" type="button">Show name <kbd>Enter</kbd></button>
<div id="answer" hidden>
<h1>{e(card["name"])}</h1>{info_html}
<form method="post" action="/submit">
<input type="hidden" name="card_id" value="{e(card_id)}">
<input type="hidden" name="reveal_ms" id="reveal_ms" value="">
{buttons}
</form>
</div>
</div>
<script>{SCRIPT}</script>""")


def done_page(stats: dict[str, int], next_due: datetime | None, now: datetime) -> bytes:
    head = ""
    if next_due is None:
        when = "There are no cards to review. Add some to cards.csv."
    else:
        if day_of(next_due) == day_of(now):
            when = f"Next card is due at {next_due:%H:%M}."
        elif day_of(next_due) == day_of(now) + timedelta(days=1):
            when = f"Next card is due tomorrow at {next_due:%H:%M}."
        else:
            when = f"Next card is due on {next_due:%Y-%m-%d}."
        wait = (next_due - now).total_seconds()
        if wait < 3600:  # reload by itself when a learning card comes due
            head = f'<meta http-equiv="refresh" content="{int(wait) + 1}">\n'
    return page(f"{stats_html(stats)}\n<h1>All done for now</h1>\n<p>{when}</p>", head)


# --- HTTP server ------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    app: App

    def send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            return self.send(204)
        if self.path.split("?")[0] != "/":
            return self.send(404, b"Not found", {"Content-Type": "text/plain"})
        app = self.app
        with app.lock:
            app.refresh_cards()
            now = now_local()
            card_id, stats, next_due = app.queue(now)
            if card_id is None:
                body = done_page(stats, next_due, now)
            else:
                intervals = preview(app.states[card_id], now, card_id)
                body = card_page(card_id, app.cards[card_id], stats, intervals)
        self.send(200, body, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"})

    def do_POST(self) -> None:
        if self.path != "/submit":
            return self.send(404, b"Not found", {"Content-Type": "text/plain"})
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode())
        card_id = form.get("card_id", [""])[0]
        grade = form.get("grade", [""])[0]
        reveal_ms = form.get("reveal_ms", [""])[0]
        if not reveal_ms.isdigit():
            reveal_ms = ""
        app = self.app
        with app.lock:
            if card_id not in app.cards or grade not in GRADES:
                return self.send(400, b"Bad request: unknown card or grade", {"Content-Type": "text/plain"})
            app.record(card_id, grade, reveal_ms, now_local())
        self.send(303, headers={"Location": "/"})


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Spaced-repetition flashcards for names and faces.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cards", type=Path, default=here / "cards.csv")
    parser.add_argument("--reviews", type=Path, default=here / "reviews.csv")
    args = parser.parse_args()

    Handler.app = App(args.cards, args.reviews)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FlashFace running at http://{args.host}:{args.port}/  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
