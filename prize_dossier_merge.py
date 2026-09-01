#!/usr/bin/env python3
"""Merge ChatGPT web-reviewed evidence into Prize Hunter Stage 2 dossiers.

The automated GitHub runner cannot always read Kaggle's JavaScript-heavy public
competition pages before entry. This merge step applies a checked evidence cache
written from live public-web research, while preserving the automatically measured
Stage 1/Stage 2 signals. Entry and submission remain approval-gated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "prize-hunter" / "dossiers"
CACHE = ROOT / "prize-hunter" / "external-research.json"
LATEST = OUT / "latest.json"


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recompute(d: dict[str, Any]) -> None:
    stage1 = float(d.get("stage1_score") or 0)
    compute = float(d.get("compute_fit") or 0)
    time_fit = float(d.get("time_fit") or 0)
    crowd_fit = float(d.get("crowd_fit") or 0)
    data_fit = float(d.get("data_fit") or 0)
    doc_fit = float(d.get("documentation_fit") or 0)
    dep = float(d.get("dependency_penalty") or 0)
    elig = float(d.get("eligibility_penalty") or 0)

    final = (
        stage1 * 0.25
        + compute * 0.25
        + time_fit * 0.15
        + crowd_fit * 0.15
        + data_fit * 0.10
        + doc_fit * 0.10
        - dep
        - elig
    )
    d["final_score"] = round(max(0.0, min(100.0, final)), 1)

    if d.get("eligibility_review"):
        d["verdict"] = "REVIEW"
    elif d.get("dependency_penalty", 0) and d["final_score"] < 60:
        d["verdict"] = "NO-GO"
    elif d.get("dependency_penalty", 0):
        d["verdict"] = "MAYBE"
    elif d["final_score"] >= 70:
        d["verdict"] = "GO"
    elif d["final_score"] >= 50:
        d["verdict"] = "MAYBE"
    else:
        d["verdict"] = "NO-GO"


def detail_md(d: dict[str, Any]) -> str:
    total = int(d.get("total_data_bytes") or 0)
    gb = total / 1024**3 if total else 0
    flags = d.get("flags") or []
    reasons = d.get("reasons") or []
    pages = d.get("public_pages_read") or d.get("pages_found") or []
    lines = [
        f"# Prize Dossier — {d['slug']}", "",
        f"**Verdict:** {d['verdict']}",
        f"**Final suitability:** {d['final_score']:.1f}/100",
        f"**Stage 1 score:** {float(d.get('stage1_score') or 0):.1f}/100",
        f"**Prize:** {d.get('reward', '')}",
        f"**Teams:** {d.get('teams') or 'unknown'}",
        f"**Days left:** {float(d.get('days_left') or 0):.0f}",
        f"**Task type:** {d.get('task_type', 'unknown')}", "",
        "## Feasibility", "",
        f"- Compute fit: **{float(d.get('compute_fit') or 0):.0f}/100**",
        f"- Time fit: **{float(d.get('time_fit') or 0):.0f}/100**",
        f"- Crowd fit: **{float(d.get('crowd_fit') or 0):.0f}/100**",
        f"- Data fit: **{float(d.get('data_fit') or 0):.0f}/100**",
        f"- Documentation confidence: **{float(d.get('documentation_fit') or 0):.0f}/100**",
        f"- Public pages read by runner: **{', '.join(pages) if pages else 'none'}**",
        f"- Competition files visible: **{int(d.get('files_count') or 0)}**",
        f"- Visible data size: **{gb:.2f} GB**" if total else "- Visible data size: **unknown / not exposed**",
        f"- Web-reviewed evidence applied: **{'yes' if d.get('web_reviewed') else 'no'}**",
        "",
    ]
    if flags:
        lines += ["## Flags", ""] + [f"- ⚠️ {x}" for x in flags] + [""]
    lines += ["## Reasons", ""] + [f"- {r}" for r in reasons] + [
        "",
        "## Entry gate", "",
        "This dossier is analysis only. Joining, accepting rules, or submitting remains approval-gated. Any age, residency, guardian-consent, licensing, team, or other eligibility requirement must be satisfied through the official competition process; Prize Hunter never bypasses it.",
        "",
    ]
    return "\n".join(lines)


def summary_md(ds: list[dict[str, Any]], cache_date: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ordered = sorted(ds, key=lambda d: float(d.get("final_score") or 0), reverse=True)
    lines = [
        "# Kaggle Prize Hunter — Stage 2 Dossiers", "",
        f"Generated: **{now}**",
        f"Web-reviewed evidence cache: **{cache_date or 'none'}**", "",
        "Stage 2 combines automated Kaggle metadata/compute analysis with previously verified public-rule research.", "",
        "| # | Verdict | Final | Stage 1 | Competition | Prize | Task | Key flag |",
        "|---:|---|---:|---:|---|---:|---|---|",
    ]
    for i, d in enumerate(ordered, 1):
        flags = d.get("flags") or []
        flag = flags[0] if flags else "none"
        lines.append(
            f"| {i} | **{d.get('verdict','MAYBE')}** | **{float(d.get('final_score') or 0):.1f}** | "
            f"{float(d.get('stage1_score') or 0):.1f} | `{d.get('slug','')}` | {d.get('reward','')} | "
            f"{d.get('task_type','unknown')} | {flag} |"
        )
    if ordered:
        lead = ordered[0]
        lines += ["", "## Best current target", "", f"**{lead['slug']}** — {lead['verdict']}, **{lead['final_score']:.1f}/100**.", ""]
    lines += [
        "## Meaning of REVIEW", "",
        "REVIEW means the opportunity may be technically interesting but an official eligibility or participation condition must be resolved before Prize Hunter can recommend entry. It is not a workaround signal.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not LATEST.exists() or not CACHE.exists():
        print("No dossier or external research cache to merge.")
        return 0

    payload = json.loads(LATEST.read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    research = cache.get("competitions") or {}
    ds = payload.get("dossiers") or []

    for d in ds:
        slug = str(d.get("slug") or "")
        extra = research.get(slug)
        if not extra:
            continue

        d["web_reviewed"] = True
        d["web_reviewed_at"] = cache.get("updated_at")
        if extra.get("task_type"):
            d["task_type"] = extra["task_type"]
        if extra.get("compute_fit") is not None:
            d["compute_fit"] = float(extra["compute_fit"])
        if extra.get("dependency"):
            d["dependency_penalty"] = max(float(d.get("dependency_penalty") or 0), 25.0)
        if extra.get("eligibility_review"):
            d["eligibility_review"] = True
            d["eligibility_penalty"] = max(float(d.get("eligibility_penalty") or 0), 8.0)
        d["flags"] = unique((d.get("flags") or []) + (extra.get("flags") or []))
        d["reasons"] = unique((d.get("reasons") or []) + (extra.get("reasons") or []))
        recompute(d)

    payload["dossiers"] = ds
    payload["external_research_updated_at"] = cache.get("updated_at")
    LATEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for d in ds:
        (OUT / f"{d['slug']}.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        (OUT / f"{d['slug']}.md").write_text(detail_md(d) + "\n", encoding="utf-8")

    (OUT / "latest.md").write_text(summary_md(ds, str(cache.get("updated_at") or "")) + "\n", encoding="utf-8")
    print(f"Merged web-reviewed evidence into {sum(1 for d in ds if d.get('web_reviewed'))} dossiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
