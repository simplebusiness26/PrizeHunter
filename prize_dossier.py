#!/usr/bin/env python3
"""Stage 2 Kaggle Prize Hunter dossier agent.

Takes the Stage 1 shortlist and inspects the top competitions using Kaggle's own
competition pages/files/topics interfaces. Produces a feasibility dossier and a
second-stage GO/MAYBE/NO-GO ranking. It never joins, accepts rules, or submits.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HUNTER_DIR = ROOT / "prize-hunter"
DOSSIER_DIR = HUNTER_DIR / "dossiers"
SOURCE = HUNTER_DIR / "latest.json"

HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)")

DEPENDENCY_PATTERNS = [
    r"required to enter",
    r"required to participate",
    r"must participate",
    r"must enter",
    r"must join",
    r"team must match",
    r"must match the team",
    r"to be eligible.*must",
    r"submission.*provided for either",
]
ELIGIBILITY_PATTERNS = [
    r"age of majority",
    r"at least 18",
    r"18 years of age",
    r"legal age",
    r"resident of",
    r"residents of",
    r"not eligible",
    r"ineligible",
]


@dataclass
class Dossier:
    slug: str
    stage1_score: float
    reward: str
    teams: int
    days_left: float
    task_type: str
    compute_fit: float
    time_fit: float
    crowd_fit: float
    data_fit: float
    documentation_fit: float
    dependency_penalty: float
    eligibility_penalty: float
    final_score: float
    verdict: str
    files_count: int
    total_data_bytes: int
    pages_found: list[str]
    flags: list[str]
    reasons: list[str]


def run(args: list[str], timeout: int = 60) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def parse_csv_output(raw: str) -> list[dict[str, str]]:
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        candidate = line.lstrip("\ufeff")
        if "," in candidate and not candidate.lower().startswith("warning:"):
            header_idx = i
            lines[i] = candidate
            break
    if header_idx is None:
        return []
    try:
        return [dict(r) for r in csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))]
    except Exception:
        return []


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = HTML_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return SPACE_RE.sub(" ", text).strip()


def fetch_page(slug: str, page_name: str) -> str:
    # Newer Kaggle CLI versions support competition pages directly.
    code, out = run([
        "kaggle", "competitions", "pages", slug,
        "--page-name", page_name, "--content", "-v", "-q"
    ])
    if code != 0:
        return ""
    rows = parse_csv_output(out)
    if not rows:
        return clean_text(out)
    row = rows[0]
    # Be tolerant of field-name changes between Kaggle CLI versions.
    for key in ("content", "body", "text", "description"):
        if row.get(key):
            return clean_text(row[key])
    return clean_text(" ".join(str(v or "") for v in row.values()))


def fetch_competition_pages(slug: str) -> tuple[dict[str, str], bool]:
    pages: dict[str, str] = {}
    supported = True
    for name in ("description", "evaluation", "rules", "data-description", "prizes"):
        text = fetch_page(slug, name)
        if text:
            pages[name] = text
    if not pages:
        # Distinguish 'command unsupported' from competition simply lacking pages.
        code, out = run(["kaggle", "competitions", "pages", "--help"])
        supported = code == 0 and "page-name" in out
    return pages, supported


def fetch_files(slug: str) -> list[dict[str, str]]:
    code, out = run([
        "kaggle", "competitions", "files", slug,
        "--page-size", "200", "-v", "-q"
    ])
    if code != 0:
        return []
    return parse_csv_output(out)


def fetch_topics(slug: str) -> list[dict[str, str]]:
    code, out = run([
        "kaggle", "competitions", "topics", "list", slug,
        "-s", "top", "--page-size", "10", "-v", "-q"
    ])
    if code != 0:
        return []
    return parse_csv_output(out)


def parse_size(value: Any) -> int:
    text = str(value or "").strip().lower().replace(",", "")
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        pass
    match = re.search(r"([0-9.]+)\s*(kb|mb|gb|tb|b)?", text)
    if not match:
        return 0
    number = float(match.group(1))
    unit = match.group(2) or "b"
    mult = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}[unit]
    return int(number * mult)


def infer_task(text: str, slug: str) -> tuple[str, float, list[str]]:
    t = (text + " " + slug).lower()
    reasons: list[str] = []

    if any(k in t for k in ("paper track", "writeup", "write-up", "project report", "essay")):
        reasons.append("documentation/paper style task")
        return "paper/writeup", 85.0, reasons
    if any(k in t for k in ("video generation", "3d", "neural rendering")):
        reasons.append("heavy generative/3D workload")
        return "heavy generative", 30.0, reasons
    if any(k in t for k in ("arc-agi", "abstraction and reasoning", "generalization", "reasoning corpus")):
        reasons.append("frontier reasoning research")
        return "reasoning/research", 40.0, reasons
    if any(k in t for k in ("simulation", "agent", "gameplay", "battle", "environment")):
        reasons.append("agent/simulation workload")
        return "agent/simulation", 60.0, reasons
    if any(k in t for k in ("large language", "llm", "language model", "transformer", "nlp", "text classification")):
        reasons.append("language-model workload")
        return "NLP/LLM", 58.0, reasons
    if any(k in t for k in ("image", "x-ray", "mri", "ct scan", "detection", "segmentation", "vision", "cell tracking")):
        reasons.append("computer-vision workload")
        return "computer vision", 67.0, reasons
    if any(k in t for k in ("tabular", "regression", "forecast", "classification", "csv", "time series")):
        reasons.append("conventional ML workload")
        return "tabular/ML", 90.0, reasons

    reasons.append("task type unclear from available documentation")
    return "unknown", 55.0, reasons


def score_time(days: float) -> float:
    if days >= 45:
        return 90.0
    if days >= 21:
        return 80.0
    if days >= 10:
        return 65.0
    if days >= 5:
        return 40.0
    return 15.0


def score_crowd(teams: int) -> float:
    if teams <= 0:
        return 50.0
    # Smoothly penalise very crowded competitions.
    return max(10.0, min(95.0, 100.0 - 22.0 * math.log10(teams + 1)))


def score_data(files: list[dict[str, str]], task_type: str) -> tuple[float, int, list[str]]:
    reasons: list[str] = []
    total = 0
    for row in files:
        size = 0
        for key in ("size", "totalBytes", "bytes", "fileSize"):
            if row.get(key):
                size = parse_size(row[key])
                if size:
                    break
        total += size

    if task_type == "paper/writeup" and not files:
        reasons.append("no competition dataset needed for paper-style task")
        return 85.0, total, reasons
    if not files:
        reasons.append("competition files not accessible before rules acceptance or not exposed")
        return 45.0, total, reasons
    if total == 0:
        reasons.append(f"{len(files)} files visible; size metadata unavailable")
        return 70.0, total, reasons
    gb = total / (1024**3)
    if gb <= 5:
        reasons.append(f"dataset footprint about {gb:.1f} GB")
        return 90.0, total, reasons
    if gb <= 20:
        reasons.append(f"dataset footprint about {gb:.1f} GB")
        return 75.0, total, reasons
    if gb <= 50:
        reasons.append(f"large dataset footprint about {gb:.1f} GB")
        return 50.0, total, reasons
    reasons.append(f"very large dataset footprint about {gb:.1f} GB")
    return 25.0, total, reasons


def pattern_hits(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, lower, flags=re.I | re.S):
            hits.append(pattern)
    return hits


def make_dossier(item: dict[str, Any]) -> Dossier:
    slug = str(item.get("ref") or "").strip().rstrip("/").split("/")[-1]
    stage1 = float(item.get("score") or 0)
    reward = str(item.get("reward") or "")
    teams = int(item.get("team_count") or 0)
    days = float(item.get("days_left") or 0)

    pages, pages_supported = fetch_competition_pages(slug)
    files = fetch_files(slug)
    topics = fetch_topics(slug)
    all_text = " ".join(pages.values())

    task_type, compute_fit, task_reasons = infer_task(all_text, slug)
    time_fit = score_time(days)
    crowd_fit = score_crowd(teams)
    data_fit, total_bytes, data_reasons = score_data(files, task_type)
    documentation_fit = 95.0 if len(pages) >= 3 else 75.0 if pages else 50.0

    flags: list[str] = []
    reasons = task_reasons + data_reasons

    dependency_hits = pattern_hits(all_text, DEPENDENCY_PATTERNS)
    dependency_penalty = 0.0
    if dependency_hits:
        dependency_penalty = 22.0
        flags.append("hidden prerequisite/dependency detected")
        reasons.append("rules/overview appear to require participation or a submission in another track/competition")

    eligibility_hits = pattern_hits(all_text, ELIGIBILITY_PATTERNS)
    eligibility_penalty = 0.0
    if eligibility_hits:
        eligibility_penalty = 8.0
        flags.append("eligibility terms require manual review")
        reasons.append("age/residency/eligibility language detected; do not enter until verified")

    if not pages_supported:
        flags.append("installed Kaggle CLI cannot expose competition pages")
        reasons.append("documentation depth limited by current CLI version")
        documentation_fit = min(documentation_fit, 40.0)
    elif not pages:
        flags.append("competition pages unavailable through CLI")
        reasons.append("rules/overview could not be inspected automatically")

    if topics:
        reasons.append(f"reviewed metadata for {len(topics)} top discussion topics")

    final = (
        stage1 * 0.25
        + compute_fit * 0.25
        + time_fit * 0.15
        + crowd_fit * 0.15
        + data_fit * 0.10
        + documentation_fit * 0.10
        - dependency_penalty
        - eligibility_penalty
    )
    final = round(max(0.0, min(100.0, final)), 1)

    if eligibility_hits:
        verdict = "REVIEW"
    elif final >= 70 and not dependency_hits:
        verdict = "GO"
    elif final >= 50:
        verdict = "MAYBE"
    else:
        verdict = "NO-GO"

    return Dossier(
        slug=slug,
        stage1_score=stage1,
        reward=reward,
        teams=teams,
        days_left=days,
        task_type=task_type,
        compute_fit=round(compute_fit, 1),
        time_fit=round(time_fit, 1),
        crowd_fit=round(crowd_fit, 1),
        data_fit=round(data_fit, 1),
        documentation_fit=round(documentation_fit, 1),
        dependency_penalty=dependency_penalty,
        eligibility_penalty=eligibility_penalty,
        final_score=final,
        verdict=verdict,
        files_count=len(files),
        total_data_bytes=total_bytes,
        pages_found=sorted(pages.keys()),
        flags=flags,
        reasons=reasons,
    )


def dossier_markdown(d: Dossier) -> str:
    data_gb = d.total_data_bytes / (1024**3) if d.total_data_bytes else 0
    lines = [
        f"# Prize Dossier — {d.slug}", "",
        f"**Stage 2 verdict:** {d.verdict}",
        f"**Final suitability:** {d.final_score:.1f}/100",
        f"**Stage 1 opportunity score:** {d.stage1_score:.1f}/100",
        f"**Prize:** {d.reward}",
        f"**Teams:** {d.teams if d.teams > 0 else 'unknown'}",
        f"**Days left:** {d.days_left:.0f}",
        f"**Task:** {d.task_type}", "",
        "## Feasibility", "",
        f"- Compute fit for our Kaggle T4 environment: **{d.compute_fit:.0f}/100**",
        f"- Time fit: **{d.time_fit:.0f}/100**",
        f"- Crowd fit: **{d.crowd_fit:.0f}/100**",
        f"- Data fit: **{d.data_fit:.0f}/100**",
        f"- Documentation confidence: **{d.documentation_fit:.0f}/100**",
        f"- Competition files visible: **{d.files_count}**",
        f"- Visible data size: **{data_gb:.2f} GB**" if d.total_data_bytes else "- Visible data size: **unknown / not exposed**",
        f"- Pages inspected: **{', '.join(d.pages_found) if d.pages_found else 'none'}**", "",
    ]
    if d.flags:
        lines += ["## Flags", ""] + [f"- ⚠️ {x}" for x in d.flags] + [""]
    lines += ["## Why", ""] + [f"- {x}" for x in d.reasons] + [
        "",
        "## Gate",
        "",
        "GO means worth pursuing further — it is **not** permission to join, accept rules, or submit. Eligibility and official rules must still be checked before entry.",
        "",
    ]
    return "\n".join(lines)


def summary_markdown(ds: list[Dossier]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Kaggle Prize Hunter — Stage 2 Dossiers", "",
        f"Generated: **{now}**", "",
        "Stage 2 reranks the Stage 1 shortlist for practical feasibility on our current Kaggle setup.", "",
        "| # | Verdict | Final | Stage 1 | Competition | Prize | Task | Key flag |",
        "|---:|---|---:|---:|---|---:|---|---|",
    ]
    ordered = sorted(ds, key=lambda d: d.final_score, reverse=True)
    for i, d in enumerate(ordered, 1):
        flag = d.flags[0] if d.flags else "none"
        lines.append(f"| {i} | **{d.verdict}** | **{d.final_score:.1f}** | {d.stage1_score:.1f} | `{d.slug}` | {d.reward} | {d.task_type} | {flag} |")
    if ordered:
        lead = ordered[0]
        lines += ["", "## Best current target", "", f"**{lead.slug}** — {lead.verdict}, **{lead.final_score:.1f}/100**.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if not SOURCE.exists():
        raise RuntimeError("prize-hunter/latest.json is missing; run Stage 1 first")
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = payload.get("opportunities") or []
    top = items[: max(1, min(args.top, 10))]

    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    dossiers = [make_dossier(item) for item in top]

    for d in dossiers:
        (DOSSIER_DIR / f"{d.slug}.md").write_text(dossier_markdown(d) + "\n", encoding="utf-8")
        (DOSSIER_DIR / f"{d.slug}.json").write_text(json.dumps(asdict(d), indent=2) + "\n", encoding="utf-8")

    (DOSSIER_DIR / "latest.md").write_text(summary_markdown(dossiers) + "\n", encoding="utf-8")
    (DOSSIER_DIR / "latest.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "dossiers": [asdict(d) for d in dossiers]}, indent=2) + "\n",
        encoding="utf-8",
    )

    ordered = sorted(dossiers, key=lambda d: d.final_score, reverse=True)
    print(f"Built {len(dossiers)} Stage 2 dossiers.")
    if ordered:
        print(f"Best current target: {ordered[0].slug} ({ordered[0].verdict}, {ordered[0].final_score:.1f}/100)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
