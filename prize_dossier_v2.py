#!/usr/bin/env python3
"""Kaggle Prize Hunter Stage 2 — public dossier agent.

Reads the Stage 1 shortlist, inspects public Kaggle competition pages without
joining or accepting rules, checks data/discussion metadata through the Kaggle
CLI, and produces feasibility dossiers. It never enters or submits.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
HUNTER = ROOT / "prize-hunter"
SOURCE = HUNTER / "latest.json"
OUT = HUNTER / "dossiers"

DEPENDENCY_PATTERNS = [
    r"required to enter",
    r"required to participate",
    r"participation .* required",
    r"must participate",
    r"must enter",
    r"must join",
    r"team must match",
    r"must match the team",
    r"document .* submission .* (?:arc-agi-2|arc-agi-3)",
    r"submission .* provided for either",
]
ELIGIBILITY_PATTERNS = [
    r"age of majority",
    r"at least 18",
    r"18 years of age",
    r"legal age",
    r"residents? of",
    r"excluded jurisdiction",
    r"employees? of .* (?:not eligible|may not)",
]


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", html.unescape(" ".join(self.parts))).strip()


@dataclass
class Dossier:
    slug: str
    reward: str
    teams: int
    days_left: float
    stage1_score: float
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
    public_pages_read: list[str]
    files_count: int
    total_data_bytes: int
    flags: list[str]
    reasons: list[str]


def run(args: list[str], timeout: int = 45) -> tuple[int, str]:
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    return p.returncode, p.stdout.strip()


def parse_csv(raw: str) -> list[dict[str, str]]:
    lines = raw.splitlines()
    idx = None
    for i, line in enumerate(lines):
        line = line.lstrip("\ufeff")
        if "," in line and not line.lower().startswith("warning:"):
            idx = i
            lines[i] = line
            break
    if idx is None:
        return []
    try:
        return [dict(r) for r in csv.DictReader(io.StringIO("\n".join(lines[idx:])))]
    except Exception:
        return []


def fetch_public_page(slug: str, label: str, suffix: str) -> tuple[str, str]:
    url = f"https://www.kaggle.com/competitions/{slug}/{suffix}".rstrip("/")
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AnimationFactory-PrizeHunter/1.0)",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        if r.status_code != 200 or len(r.text) < 500:
            return label, ""
        parser = TextExtractor()
        parser.feed(r.text)
        text = parser.text()
        if len(text) < 300:
            return label, ""
        return label, text[:120000]
    except Exception:
        return label, ""


def fetch_pages(slug: str) -> dict[str, str]:
    specs = [("overview", "overview/description"), ("rules", "rules"), ("data", "data")]
    pages: dict[str, str] = {}
    for label, suffix in specs:
        name, text = fetch_public_page(slug, label, suffix)
        if text:
            pages[name] = text
    return pages


def fetch_files(slug: str) -> list[dict[str, str]]:
    code, out = run(["kaggle", "competitions", "files", slug, "--page-size", "200", "-v", "-q"])
    return parse_csv(out) if code == 0 else []


def fetch_topics(slug: str) -> list[dict[str, str]]:
    code, out = run(["kaggle", "competitions", "topics", "list", slug, "-s", "top", "--page-size", "10", "-v", "-q"])
    return parse_csv(out) if code == 0 else []


def parse_size(value: Any) -> int:
    s = str(value or "").strip().lower().replace(",", "")
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        pass
    m = re.search(r"([0-9.]+)\s*(b|kb|mb|gb|tb)?", s)
    if not m:
        return 0
    n = float(m.group(1))
    unit = m.group(2) or "b"
    return int(n * {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}[unit])


def infer_task(text: str, slug: str) -> tuple[str, float, str]:
    t = (text + " " + slug).lower()
    if any(k in t for k in ("paper track", "writeup", "write-up", "project report", "paper award")):
        return "paper/writeup", 85.0, "paper/writeup deliverable"
    if any(k in t for k in ("arc-agi", "abstraction and reasoning", "novel human-solvable", "generalization")):
        return "reasoning/research", 40.0, "frontier reasoning research"
    if any(k in t for k in ("simulation", "gameplay", "battle challenge", "training agent", "agentic play")):
        return "agent/simulation", 60.0, "agent/simulation workload"
    if any(k in t for k in ("video generation", "3d", "neural rendering")):
        return "heavy generative", 30.0, "heavy generative workload"
    if any(k in t for k in ("large language", "language model", "llm", "nlp", "transformer")):
        return "NLP/LLM", 58.0, "language-model workload"
    if any(k in t for k in ("image", "mri", "x-ray", "segmentation", "object detection", "cell tracking", "computer vision")):
        return "computer vision", 67.0, "computer-vision workload"
    if any(k in t for k in ("tabular", "regression", "forecast", "time series", "classification")):
        return "tabular/ML", 90.0, "conventional ML workload"
    if any(k in t for k in ("life science", "biology", "protein", "molecule", "scientific")):
        return "scientific ML", 55.0, "scientific/research workload"
    return "unknown", 55.0, "task type not confidently classified"


def time_fit(days: float) -> float:
    if days >= 45: return 90.0
    if days >= 21: return 80.0
    if days >= 10: return 65.0
    if days >= 5: return 40.0
    return 15.0


def crowd_fit(teams: int) -> float:
    if teams <= 0: return 50.0
    return max(10.0, min(95.0, 100.0 - 22.0 * math.log10(teams + 1)))


def data_fit(files: list[dict[str, str]], task: str) -> tuple[float, int, str]:
    total = 0
    for row in files:
        for key in ("size", "totalBytes", "bytes", "fileSize"):
            if row.get(key):
                total += parse_size(row[key]); break
    if task == "paper/writeup" and not files:
        return 85.0, total, "no standalone dataset appears necessary for the writeup track"
    if not files:
        return 45.0, total, "competition files are not exposed before entry or no files are listed"
    if total == 0:
        return 70.0, total, f"{len(files)} files visible; size metadata unavailable"
    gb = total / 1024**3
    if gb <= 5: return 90.0, total, f"dataset footprint about {gb:.1f} GB"
    if gb <= 20: return 75.0, total, f"dataset footprint about {gb:.1f} GB"
    if gb <= 50: return 50.0, total, f"large dataset footprint about {gb:.1f} GB"
    return 25.0, total, f"very large dataset footprint about {gb:.1f} GB"


def has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I | re.S) for p in patterns)


def build(item: dict[str, Any]) -> Dossier:
    slug = str(item.get("ref") or "").strip().rstrip("/").split("/")[-1]
    stage1 = float(item.get("score") or 0); reward = str(item.get("reward") or "")
    teams = int(item.get("team_count") or 0); days = float(item.get("days_left") or 0)
    pages = fetch_pages(slug); files = fetch_files(slug); topics = fetch_topics(slug); text = " ".join(pages.values())
    task, compute, task_reason = infer_task(text, slug); tf = time_fit(days); cf = crowd_fit(teams)
    df, total, data_reason = data_fit(files, task); doc = 95.0 if len(pages) >= 2 else 75.0 if pages else 35.0
    flags: list[str] = []; reasons = [task_reason, data_reason]
    dep = has_pattern(text.lower(), DEPENDENCY_PATTERNS); dep_penalty = 25.0 if dep else 0.0
    if dep:
        flags.append("hard prerequisite/dependency detected"); reasons.append("public competition text requires participation or a submission in another track/competition")
    elig = has_pattern(text.lower(), ELIGIBILITY_PATTERNS); elig_penalty = 8.0 if elig else 0.0
    if elig:
        flags.append("age/residency/eligibility terms require review"); reasons.append("eligibility language detected; do not enter until the official rules are checked")
    if not pages:
        flags.append("public Kaggle pages could not be read automatically"); reasons.append("documentation confidence reduced")
    else:
        reasons.append(f"read public pages: {', '.join(sorted(pages))}")
    if topics: reasons.append(f"inspected metadata for {len(topics)} top discussion topics")
    final = stage1*0.25 + compute*0.25 + tf*0.15 + cf*0.15 + df*0.10 + doc*0.10 - dep_penalty - elig_penalty
    final = round(max(0.0, min(100.0, final)), 1)
    if elig: verdict = "REVIEW"
    elif dep and final < 60: verdict = "NO-GO"
    elif dep: verdict = "MAYBE"
    elif final >= 70: verdict = "GO"
    elif final >= 50: verdict = "MAYBE"
    else: verdict = "NO-GO"
    return Dossier(slug,reward,teams,days,stage1,task,compute,tf,round(cf,1),df,doc,dep_penalty,elig_penalty,final,verdict,sorted(pages),len(files),total,flags,reasons)


def detail_md(d: Dossier) -> str:
    gb = d.total_data_bytes / 1024**3 if d.total_data_bytes else 0
    lines=[f"# Prize Dossier — {d.slug}","",f"**Verdict:** {d.verdict}",f"**Final suitability:** {d.final_score:.1f}/100",f"**Stage 1 score:** {d.stage1_score:.1f}/100",f"**Prize:** {d.reward}",f"**Teams:** {d.teams if d.teams else 'unknown'}",f"**Days left:** {d.days_left:.0f}",f"**Task type:** {d.task_type}","","## Feasibility","",f"- Compute fit: **{d.compute_fit:.0f}/100**",f"- Time fit: **{d.time_fit:.0f}/100**",f"- Crowd fit: **{d.crowd_fit:.0f}/100**",f"- Data fit: **{d.data_fit:.0f}/100**",f"- Documentation confidence: **{d.documentation_fit:.0f}/100**",f"- Public pages read: **{', '.join(d.public_pages_read) if d.public_pages_read else 'none'}**",f"- Competition files visible: **{d.files_count}**",f"- Visible data size: **{gb:.2f} GB**" if d.total_data_bytes else "- Visible data size: **unknown / not exposed**",""]
    if d.flags: lines += ["## Flags",""]+[f"- ⚠️ {x}" for x in d.flags]+[""]
    lines += ["## Reasons",""]+[f"- {r}" for r in d.reasons]+["","## Entry gate","","This dossier is analysis only. Joining, accepting rules, or submitting remains approval-gated and must satisfy the official eligibility rules.",""]
    return "\n".join(lines)


def summary_md(ds: list[Dossier]) -> str:
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"); ordered=sorted(ds,key=lambda d:d.final_score,reverse=True)
    lines=["# Kaggle Prize Hunter — Stage 2 Dossiers","",f"Generated: **{now}**","","Stage 2 reranks the top Stage 1 opportunities using public competition pages plus Kaggle metadata.","","| # | Verdict | Final | Stage 1 | Competition | Prize | Task | Key flag |","|---:|---|---:|---:|---|---:|---|---|"]
    for i,d in enumerate(ordered,1):
        flag=d.flags[0] if d.flags else "none"; lines.append(f"| {i} | **{d.verdict}** | **{d.final_score:.1f}** | {d.stage1_score:.1f} | `{d.slug}` | {d.reward} | {d.task_type} | {flag} |")
    if ordered: lines += ["","## Best current target","",f"**{ordered[0].slug}** — {ordered[0].verdict}, **{ordered[0].final_score:.1f}/100**.",""]
    return "\n".join(lines)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--top",type=int,default=5); args=ap.parse_args()
    payload=json.loads(SOURCE.read_text(encoding="utf-8")); items=(payload.get("opportunities") or [])[:max(1,min(args.top,10))]; OUT.mkdir(parents=True,exist_ok=True)
    ds=[build(item) for item in items]
    for d in ds:
        (OUT/f"{d.slug}.md").write_text(detail_md(d)+"\n",encoding="utf-8"); (OUT/f"{d.slug}.json").write_text(json.dumps(asdict(d),indent=2)+"\n",encoding="utf-8")
    (OUT/"latest.md").write_text(summary_md(ds)+"\n",encoding="utf-8"); (OUT/"latest.json").write_text(json.dumps({"generated_at":datetime.now(timezone.utc).isoformat(),"dossiers":[asdict(d) for d in ds]},indent=2)+"\n",encoding="utf-8")
    ordered=sorted(ds,key=lambda d:d.final_score,reverse=True); print(f"Built {len(ds)} Stage 2 public dossiers.")
    if ordered: print(f"Best current target: {ordered[0].slug} ({ordered[0].verdict}, {ordered[0].final_score:.1f}/100)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
