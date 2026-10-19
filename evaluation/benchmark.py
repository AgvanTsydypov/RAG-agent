"""Correctness benchmark: baseline vs single-pass ablation vs full agent.

Unlike the RAGAS suite (run_eval.py), which scores grounding and relevancy,
this checks whether answers are RIGHT:

  numeric  - deterministic: every expected figure must appear in the answer
  grounded - deterministic: every figure in the answer traces to a tool output
  judge    - an LLM grader (a different model from the generator) compares the
             answer to the reference: correct=1, partial=0.5, incorrect=0
  tools    - agent systems only: expected tools called, forbidden ones avoided

  strict   - headline metric: judge says "correct" AND every applicable check
             above passes. pass^k: strict in ALL k runs (reliability).

Usage:
  python -m evaluation.benchmark                       # all systems, 1 run
  python -m evaluation.benchmark --runs 3              # mean +- std over runs
  python -m evaluation.benchmark --systems agent --categories needle,trend
"""

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI, RateLimitError

from app.config import settings
from evaluation.collect import SYSTEMS, Trace
from evaluation.dataset import EVAL_SET

# Judge with a different model than the one answering, to limit self-preference.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-5.5")
RESULTS_DIR = Path(__file__).parent / "results"

_client = OpenAI(api_key=settings.openai_api_key)

# --- Deterministic numeric check -------------------------------------------
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Answers state money as "$391.0 billion", "$391,035 million" or raw dollars,
# so try each scale against the reference (stored in millions).
_USD_M_SCALES = (1, 1_000, 1e-6, 1e-3)


def _numbers_in(text: str) -> list[float]:
    values = []
    for match in _NUMBER.findall(text):
        try:
            values.append(float(match.rstrip(",").replace(",", "")))
        except ValueError:
            pass
    return values


def _matches(expected: float, found: float, unit: str) -> bool:
    if unit == "pct":
        return abs(found - expected) <= 0.15  # percentage points
    if unit == "plain":
        return abs(found - expected) <= 0.011
    if unit == "usd_m_exact":  # "$220,960 million" or "$220.96 billion", no rounding
        return any(abs(found * s - expected) < 0.5 for s in _USD_M_SCALES)
    return any(
        abs(found * s - expected) <= 0.005 * abs(expected) for s in _USD_M_SCALES
    )


def numeric_check(item: dict, answer: str) -> bool | None:
    if not item.get("numbers"):
        return None
    found = _numbers_in(answer)
    return all(
        any(_matches(exp, f, item["unit"]) for f in found) for exp in item["numbers"]
    )


# --- Grounding check --------------------------------------------------------
# Every significant number in the answer must be traceable to a tool output
# (or the question itself). Tool outputs hold raw dollars ("391035000000") and
# calculator results ("0.2551..."), so an answer figure is grounded when some
# source value, at some scale, rounds to it at the precision the answer states.
# A correct figure recalled from memory still fails: that is the point.
_ANSWER_NUMBER = re.compile(r"(\$)?(-?\d[\d,]*(?:\.\d+)?)")
_GROUND_SCALES = (1, 1e-2, 1e3, 1e6, 1e9)  # as-is, fraction->pct, K, M, B
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _significant_numbers(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for figures worth checking. Skips years, dates, list
    markers and section refs ("FY2025", "June 30", "Item 1A")."""
    out = []
    for dollar, raw in _ANSWER_NUMBER.findall(text):
        raw = raw.rstrip(",")  # "September 27, 2025" -> "27", not "27,"
        try:
            value = abs(float(raw.replace(",", "")))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not dollar and decimals == 0 and "," not in raw:
            if value <= 31 or 1990 <= value <= 2035:
                continue
        out.append((value, decimals))
    return out


def _ungrounded(item: dict, trace: Trace) -> tuple[list, list] | None:
    """Split the answer's figures into (grounded, ungrounded)."""
    figures = _significant_numbers(trace.answer)
    if not figures:
        return None
    sources = " ".join(
        [*trace.contexts, item["question"], *item.get("turns", [])]
    )
    # Dates and years in sources are not figures; "2024" / 1e3 must not
    # "ground" a fabricated 2.0%.
    sources = _ISO_DATE.sub(" ", sources)
    pool = {abs(v) for v in _numbers_in(sources) if not (v.is_integer() and 1990 <= v <= 2035)}
    grounded, ungrounded = [], []
    for value, decimals in figures:
        tolerance = 0.5 * 10**-decimals + 1e-9
        hit = any(abs(c / s - value) <= tolerance for c in pool for s in _GROUND_SCALES)
        (grounded if hit else ungrounded).append((value, decimals))
    return grounded, ungrounded


def grounding_check(item: dict, trace: Trace) -> bool | None:
    """Strict: every figure is in a tool output (derived ones via the calculator,
    as the agent's own prompt requires)."""
    split = _ungrounded(item, trace)
    return None if split is None else not split[1]


def _derivable(value: float, decimals: int, inputs: list[float]) -> bool:
    tolerance = 0.5 * 10**-decimals + 1e-9
    for a in inputs:
        for b in inputs:
            if a == b:
                continue
            candidates = [a - b, a + b]
            if b:
                candidates += [a / b * 100, (a / b - 1) * 100]
            if any(abs(abs(c) - value) <= tolerance for c in candidates):
                return True
    return False


def fabrication_check(item: dict, trace: Trace) -> bool | None:
    """True if some figure is neither in a tool output nor one arithmetic step
    from grounded figures shown in the same answer. Separates invented numbers
    from correct mental arithmetic that skipped the calculator."""
    split = _ungrounded(item, trace)
    if split is None:
        return None
    inputs = [v for v, _ in split[0]]
    return any(not _derivable(v, d, inputs) for v, d in split[1])


# --- LLM judge --------------------------------------------------------------
JUDGE_PROMPT = """You grade answers from a financial research assistant that \
answers questions about Apple and Microsoft 10-K filings.

Compare the ANSWER to the REFERENCE for the QUESTION and return JSON:
{{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}

Rules:
- correct: the key facts match the reference (small rounding differences are fine).
- partial: some key facts are right but others are missing, vague or wrong.
- incorrect: wrong key facts, an invented figure, or a refusal when the reference has an answer.
- If the reference says the question has a false premise, the answer must correct
  the premise; explaining a nonexistent trend is incorrect.
- If the reference says the information is unavailable, a clear "not available"
  is correct; any invented or unlabeled figure is incorrect.
- Extra correct detail is fine; do not reward length.

CATEGORY: {category}
{history}QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}"""


def judge(item: dict, answer: str) -> dict:
    history = "".join(f"EARLIER USER TURN: {t}\n" for t in item.get("turns", []))
    prompt = JUDGE_PROMPT.format(
        category=item["category"],
        history=history,
        question=item["question"],
        reference=item["reference"],
        answer=answer or "(empty)",
    )
    response = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    verdict = parsed.get("verdict", "incorrect")
    score = {"correct": 1.0, "partial": 0.5}.get(verdict, 0.0)
    return {"verdict": verdict, "score": score, "reason": parsed.get("reason", "")}


# --- Tool-use check ---------------------------------------------------------
def tool_check(system: str, item: dict, trace: Trace) -> bool | None:
    if system == "baseline" or "expected_tools" not in item:
        return None
    called = set(trace.tools)
    return set(item["expected_tools"]) <= called and not (
        set(item.get("forbidden_tools", [])) & called
    )


# --- Runner -----------------------------------------------------------------
def evaluate_item(system: str, item: dict) -> dict:
    try:
        trace = SYSTEMS[system](item["question"], item.get("turns"))
    except RateLimitError as exc:
        if "insufficient_quota" in str(exc):
            raise  # out of credits: scoring this as a wrong answer would lie
        trace = Trace(answer=f"ERROR: {exc!r}")
    except Exception as exc:  # one broken item should not sink the whole run
        trace = Trace(answer=f"ERROR: {exc!r}")
    graded = judge(item, trace.answer)
    result = {
        "id": item["id"],
        "category": item["category"],
        "system": system,
        "judge": graded["score"],
        "verdict": graded["verdict"],
        "judge_reason": graded["reason"],
        "numeric": numeric_check(item, trace.answer),
        "grounded": grounding_check(item, trace),
        "fabricated": fabrication_check(item, trace),
        "tools_ok": tool_check(system, item, trace),
        **asdict(trace),  # contexts kept so grading can be re-run offline
    }
    checks = (result["numeric"], result["grounded"], result["tools_ok"])
    result["strict"] = graded["verdict"] == "correct" and False not in checks
    return result


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(per_run: list[float | None], pct: bool = True) -> str:
    vals = [v for v in per_run if v is not None]
    if not vals:
        return "-"
    mean = statistics.mean(vals)
    text = f"{mean * 100:.0f}%" if pct else f"{mean:.1f}"
    if len(vals) > 1:
        sd = statistics.stdev(vals)
        text += f" ±{sd * 100:.0f}" if pct else f" ±{sd:.1f}"
    return text


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report(results: list[dict], systems: list[str], runs: int) -> str:
    # per_run[system][metric] -> list of per-run means
    by = defaultdict(list)
    for r in results:
        by[(r["system"], r["run"])].append(r)

    def per_run(system, fn, rows_filter=lambda r: True):
        return [
            _mean(fn(r) for r in by[(system, run)] if rows_filter(r))
            for run in range(runs)
        ]

    def pass_k(system, rows_filter=lambda r: True):
        ok = defaultdict(list)
        for r in results:
            if r["system"] == system and rows_filter(r):
                ok[r["id"]].append(r["strict"])
        return [_mean(all(v) for v in ok.values())] if ok else [None]

    overall = [
        [
            s,
            _fmt(per_run(s, lambda r: r["judge"])),
            _fmt(per_run(s, lambda r: r["strict"])),
            _fmt(pass_k(s)) if runs > 1 else "-",
            _fmt(per_run(s, lambda r: r["numeric"])),
            _fmt(per_run(s, lambda r: r["grounded"])),
            _fmt(per_run(s, lambda r: r.get("fabricated"))),
            _fmt(per_run(s, lambda r: r["tools_ok"])),
            _fmt(per_run(s, lambda r: r["steps"]), pct=False),
            _fmt(per_run(s, lambda r: r["latency_s"]), pct=False),
        ]
        for s in systems
    ]
    out = [
        f"n={len(EVAL_SET)} items x {runs} run(s), judge={JUDGE_MODEL}, generator={settings.chat_model}\n",
        _table(
            ["system", "judge (lenient)", "STRICT", f"pass^{runs}", "numeric",
             "grounded", "fabricated", "tool use", "LLM steps", "latency s"],
            overall,
        ),
        f"\nSTRICT accuracy by category (pass^{runs} in brackets):\n",
    ]
    categories = sorted({i["category"] for i in EVAL_SET})
    counts = defaultdict(int)
    for item in EVAL_SET:
        counts[item["category"]] += 1

    def cell(s, c):
        in_c = lambda r, c=c: r["category"] == c
        text = _fmt(per_run(s, lambda r: r["strict"], in_c))
        return text + (f" [{_fmt(pass_k(s, in_c))}]" if runs > 1 else "")

    out.append(
        _table(
            ["category", "n", *systems],
            [[c, str(counts[c]), *(cell(s, c) for s in systems)] for c in categories],
        )
    )
    return "\n".join(out)


def rescore(path: Path) -> None:
    """Re-run the deterministic checks on saved traces; keeps judge verdicts."""
    items = {i["id"]: i for i in EVAL_SET}
    results = json.loads(path.read_text())
    for r in results:
        item = items[r["id"]]
        trace = Trace(**{k: r[k] for k in ("answer", "contexts", "tools", "steps", "latency_s")})
        r["numeric"] = numeric_check(item, trace.answer)
        r["grounded"] = grounding_check(item, trace)
        r["fabricated"] = fabrication_check(item, trace)
        r["tools_ok"] = tool_check(r["system"], item, trace)
        checks = (r["numeric"], r["grounded"], r["tools_ok"])
        r["strict"] = r["verdict"] == "correct" and False not in checks
    systems = list(dict.fromkeys(r["system"] for r in results))
    summary = report(results, systems, len({r["run"] for r in results}))
    print(summary)
    path.write_text(json.dumps(results, indent=2))
    path.with_suffix(".md").write_text(summary + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--categories", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rescore", help="re-grade a saved results .json offline (no API calls)")
    args = parser.parse_args()

    if args.rescore:
        rescore(Path(args.rescore))
        return

    global EVAL_SET
    if args.categories:
        wanted = set(args.categories.split(","))
        EVAL_SET = [i for i in EVAL_SET if i["category"] in wanted]
    systems = args.systems.split(",")

    jobs = [(run, s, item) for run in range(args.runs) for s in systems for item in EVAL_SET]
    print(f"Running {len(jobs)} (run, system, item) jobs...")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Append each result as it lands, so a crash or empty quota keeps the work.
    partial = RESULTS_DIR / f"{stamp}.jsonl"
    lock, out_of_credits = threading.Lock(), threading.Event()

    def work(job):
        run, system, item = job
        if out_of_credits.is_set():
            return None
        try:
            result = {"run": run, **evaluate_item(system, item)}
        except RateLimitError:
            out_of_credits.set()
            return None
        mark = "+" if result["strict"] else "x"
        print(f"  [{mark}] run{run} {system:<11} {item['id']}", flush=True)
        with lock, partial.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(work, jobs) if r is not None]

    runs = args.runs
    if out_of_credits.is_set():
        # Report only runs that finished for every system and item.
        done = defaultdict(int)
        for r in results:
            done[r["run"]] += 1
        complete = [k for k in range(args.runs) if done[k] == len(systems) * len(EVAL_SET)]
        print(f"\nOUT OF CREDITS after {len(results)}/{len(jobs)} jobs; complete runs: {complete}")
        results = [r for r in results if r["run"] in complete]
        for r in results:
            r["run"] = complete.index(r["run"])
        runs = len(complete)
        if not runs:
            return

    summary = report(results, systems, runs)
    print("\n" + summary)

    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / f"{stamp}.md").write_text(summary + "\n")
    print(f"\nSaved per-item results to {RESULTS_DIR / f'{stamp}.json'}")


if __name__ == "__main__":
    main()
