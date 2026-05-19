"""
analyse.py — Document structural analyser  [Phase 2 / spec Step 1]
==================================================================
Profiles a .docx, asks Mistral to recommend a chunking strategy, and on
human approval writes config.json (read later by review_chunks.py / ingest.py).

    CLI:
      python analyse.py data/cv.docx          # analyse + print + y/n prompt
      python analyse.py data/cv.docx --yes    # analyse + write config.json, no prompt

ARCHITECTURAL DECISION: profile the document, do not assume its structure.
A Word document's *visual* hierarchy and its *underlying* markup are often two
different things — authors mix heading styles with plain direct formatting.
So this script does not hardcode "Heading 3 = company". It enumerates every
distinct formatting fingerprint and lets a human + Mistral map fingerprints to
roles. That makes the analyser reusable for any document type, not just this CV.
See LEARNING_NOTES.md, "Phase 2 — real documents lie about their structure".
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loaders import load_docx
from mistral_helpers import get_client, call_with_retry

# config.json lands here. The 4 collection names are fixed by spec Decision 4
# (a ChromaDB collection's distance metric is immutable once created).
CONFIG_PATH = "config.json"
COLLECTIONS = {
    "A":  ["cv_role_cosine", "cv_role_l2"],
    "A2": ["cv_bullet_cosine", "cv_bullet_l2"],
}

# Model for the one-off recommendation call. ARCHITECTURAL DECISION: large.
# This is meta-reasoning (reason about structure, recommend a strategy) — the
# one place large earns its cost — and it is a single call, so spend is trivial.
RECOMMEND_MODEL = "mistral-large-latest"

# Document loading (load_docx) lives in loaders/docx_loader.py — it returns the
# common Paragraph model (models/paragraph.py), shared with chunker.py so the
# preview and the ingested chunks come from an identical reading of the document.


# ──────────────────────────────────────────────
# STEP 1 — profile the document structure
# ──────────────────────────────────────────────

def build_profile(records):
    """Group paragraphs by formatting fingerprint (style, size, bold, is_list).

    This is the document-agnostic core of the analyser: it describes the
    document by what formatting actually occurs, with counts and a sample,
    instead of assuming any particular structure.
    """
    groups = defaultdict(list)
    for rec in records:
        fp = (rec.style_name, rec.rendered_size, rec.is_bold, rec.has_num_pr)
        groups[fp].append(rec)

    profile = []
    for (style, size, bold, is_list), recs in groups.items():
        profile.append({
            "style": style, "size": size, "bold": bold, "is_list": is_list,
            "count": len(recs),
            "sample": recs[0].text[:70],
        })
    profile.sort(key=lambda p: p["count"], reverse=True)
    return profile


def consistency_report(records):
    """Objective flags showing where visual hierarchy is NOT style-driven.

    Every check is factual — no guessing. Together they answer the question a
    client should ask before trusting a RAG pipeline on their documents:
    "can a style-based parser trust this document?"
    """
    flags = []

    # C1 — bullets under more than one paragraph style
    bullet_styles = sorted({r.style_name for r in records if r.has_num_pr})
    if len(bullet_styles) > 1:
        flags.append(
            f"Bullets appear under {len(bullet_styles)} different styles "
            f"({', '.join(bullet_styles)}) — detect bullets by the numPr "
            f"element, not by style name."
        )

    # 'body' size = the most common size among non-list paragraphs
    body_sizes = Counter(r.rendered_size for r in records
                         if not r.has_num_pr and r.rendered_size is not None)
    body_size = body_sizes.most_common(1)[0][0] if body_sizes else None

    # C2 — visually prominent text that carries NO heading style
    if body_size is not None:
        prominent = [r for r in records
                     if not r.has_num_pr
                     and not r.style_name.startswith("Heading")
                     and r.rendered_size is not None
                     and r.rendered_size > body_size]
        if prominent:
            samples = ", ".join(repr(r.text[:25]) for r in prominent[:3])
            flags.append(
                f"{len(prominent)} paragraph(s) are larger than body text "
                f"({body_size}pt) but use NO heading style ({samples}) — a "
                f"style-only parser would miss these as structural boundaries."
            )

    # C3 — heading styles overridden by a direct font size
    overridden = [r for r in records
                  if r.style_name.startswith("Heading") and r.override]
    if overridden:
        flags.append(
            f"{len(overridden)} heading-styled paragraph(s) carry a direct "
            f"font-size override — heading level is not a reliable size proxy."
        )

    return flags, body_size


def detect_sections(records):
    """Detect sections via the top heading style present.

    Returns (sections, section_style). Each section: title, paragraphs,
    bullets, words.
    """
    heading_styles = sorted({r.style_name for r in records
                             if r.style_name.startswith("Heading")})
    if not heading_styles:
        return [], None
    section_style = heading_styles[0]  # 'Heading 1' sorts before 'Heading 3'

    sections, current = [], None
    for rec in records:
        if rec.style_name == section_style:
            current = {"title": rec.text, "paragraphs": 0,
                       "bullets": 0, "words": 0}
            sections.append(current)
        elif current is not None:
            current["paragraphs"] += 1
            current["words"] += len(rec.text.split())
            if rec.has_num_pr:
                current["bullets"] += 1
    return sections, section_style


def estimate_chunks(records, sections):
    """Rough chunk counts for Option A and Option A.2 (estimates, labelled).

    Precision is not the point — analyse.py informs a recommendation; the
    real chunker is chunker.py (Phase 3), built from config.json.
    """
    n_bullets = sum(1 for r in records if r.has_num_pr)
    # A.2 ≈ one chunk per bullet + one per non-bullet section
    n_no_bullet_sections = sum(1 for s in sections if s["bullets"] == 0)
    a2 = n_bullets + n_no_bullet_sections
    # A ≈ one chunk per section + one per sub-heading (heading styles below
    # the top-level section style — i.e. roles/sub-sections)
    heading_styles = sorted({r.style_name for r in records
                             if r.style_name.startswith("Heading")})
    sub_styles = set(heading_styles[1:])
    n_sub = sum(1 for r in records if r.style_name in sub_styles)
    a = len(sections) + n_sub
    return {"A": a, "A2": a2, "n_bullets": n_bullets}


# ──────────────────────────────────────────────
# STEP 2 — ask Mistral for a chunking recommendation
# ──────────────────────────────────────────────

RECOMMENDATION_SHAPE = """{
  "recommended_strategy": "A" or "A2",
  "reasoning": "one short paragraph",
  "metadata_fields": ["field_name", ...],
  "prefix_template": "a string using {field_name} placeholders",
  "risks": ["risk", ...]
}"""


def ask_mistral(client, profile, flags, sections, estimates):
    """Send the structural profile to Mistral.

    Returns (recommendation, usage, prompt, raw_response) — the prompt and raw
    response are returned too so --trace can record exactly what was sent and
    received.

    ARCHITECTURAL DECISION: ask for a JSON object (response_format).
    The recommendation feeds a machine-readable config.json, so we request
    structured output rather than parsing free text. analyse.py owns the
    structural FACTS; Mistral owns the JUDGEMENT call (which strategy, risks).
    """
    summary = {
        "fingerprint_profile": profile,
        "consistency_flags": flags,
        "sections": sections,
        "estimated_chunks": estimates,
    }
    prompt = f"""You are helping configure a RAG document-ingestion pipeline.

Below is a STRUCTURAL PROFILE of a document, produced by a parser. It lists
every distinct formatting fingerprint (paragraph style, font size, bold,
list-or-not) with counts and a sample line, plus objective consistency flags.

STRUCTURAL PROFILE
{json.dumps(summary, indent=2)}

Two chunking strategies are on the table:
- Strategy A  — one chunk per role/section: full context per chunk, but less
  precise retrieval for narrow questions.
- Strategy A2 — one chunk per bullet, each carrying a context prefix:
  precise retrieval, but weaker for questions needing synthesis across a role.

The metadata_fields you recommend must be SEMANTIC fields a user would filter
or cite answers by — for a CV: company, job title, dates, section name. Do NOT
recommend formatting attributes (font size, style, bold); those are parser
internals, not user-facing facts. The prefix_template must use only those
semantic {{placeholder}} fields.

Recommend which strategy fits this document, the metadata to attach to each
chunk, a prefix template, and the main risks. Respond ONLY with a JSON object
of exactly this shape:
{RECOMMENDATION_SHAPE}
"""
    response = call_with_retry(
        client.chat.complete,
        model=RECOMMEND_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw), response.usage, prompt, raw


# ──────────────────────────────────────────────
# STEP 3 — printing + config assembly
# ──────────────────────────────────────────────

def format_analysis(records, profile, flags, sections, section_style, estimates):
    """Render the structural analysis as a text block (printed + trace file)."""
    bar = "─" * 64
    out = [bar, "STRUCTURAL ANALYSIS", bar]
    out.append(f"Paragraphs (non-empty): {len(records)}  "
               f"| in tables: {sum(r.in_table for r in records)}")

    out.append(f"\nFORMATTING-FINGERPRINT PROFILE  ({len(profile)} distinct)")
    out.append(f"  {'count':>5}  {'style':18s} {'size':>6} {'bold':>4} "
               f"{'list':>4}  sample")
    for p in profile:
        size = "-" if p["size"] is None else f"{p['size']:g}"
        out.append(f"  {p['count']:>5}  {p['style'][:18]:18s} {size:>6} "
                   f"{int(p['bold']):>4} {int(p['is_list']):>4}  {p['sample'][:34]}")

    out.append(f"\nCONSISTENCY REPORT  ({len(flags)} flag(s))")
    if not flags:
        out.append("  none — visual hierarchy is cleanly style-driven.")
    for f in flags:
        out.append(f"  ! {f}")

    out.append(f"\nSECTIONS  (detected via '{section_style}')")
    for s in sections:
        avg = s["words"] / s["paragraphs"] if s["paragraphs"] else 0
        out.append(f"  • {s['title'][:34]:34s} "
                   f"paras={s['paragraphs']:>2} bullets={s['bullets']:>2} "
                   f"words={s['words']:>3} (avg {avg:.0f}/para)")

    out.append("\nESTIMATED CHUNK COUNTS  (rough — informs the recommendation)")
    out.append(f"  Option A  (per role/section): ~{estimates['A']}")
    out.append(f"  Option A2 (per bullet)      : ~{estimates['A2']}  "
               f"({estimates['n_bullets']} bullets)")
    return "\n".join(out)


def format_recommendation(recommendation, usage):
    """Render Mistral's recommendation as a text block (printed + trace file)."""
    bar = "─" * 64
    out = [bar, "MISTRAL RECOMMENDATION", bar]
    out.append(f"Recommended strategy : {recommendation.get('recommended_strategy')}")
    out.append(f"Reasoning            : {recommendation.get('reasoning')}")
    out.append(f"Metadata fields      : {recommendation.get('metadata_fields')}")
    out.append(f"Prefix template      : {recommendation.get('prefix_template')}")
    out.append("Risks:")
    for r in recommendation.get("risks", []):
        out.append(f"  - {r}")
    out.append(f"\nTokens: {usage.prompt_tokens} prompt + "
               f"{usage.completion_tokens} completion = {usage.total_tokens}")
    return "\n".join(out)


def build_config(recommendation, section_style, flags):
    """Assemble config.json: structural facts (ours) + judgement (Mistral's)."""
    return {
        "strategy": recommendation.get("recommended_strategy", "A"),
        "document_type": "table_based",
        "section_signal": section_style,
        "bullet_signal": "paragraph has a numPr element (not a style name)",
        "decode_note": ("company / sub-section headers are detected by font "
                        "size, not by style — see consistency_flags"),
        "collections": COLLECTIONS,
        "metadata_fields": recommendation.get(
            "metadata_fields", ["company", "title", "dates", "section_type"]),
        "prefix_template": recommendation.get(
            "prefix_template", "[{company} | {title} | {dates}]"),
        "consistency_flags": flags,
    }


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a .docx and recommend a chunking strategy.")
    parser.add_argument("document", help="path to the .docx file")
    parser.add_argument("--yes", action="store_true",
                        help="write config.json without the interactive prompt")
    parser.add_argument("--trace", metavar="PATH", nargs="?",
                        const="outputs/analyse_trace.txt", default=None,
                        help="write a full trace (analysis + the exact prompt "
                             "sent to Mistral + the raw response) to PATH "
                             "(default: outputs/analyse_trace.txt)")
    args = parser.parse_args()

    path = Path(args.document)
    if not path.exists():
        sys.exit(f"Document not found: {path}")

    print(f"\nAnalysing: {path}")
    records = load_docx(path)
    if not records:
        sys.exit("No content found in the document.")

    profile = build_profile(records)
    flags, _ = consistency_report(records)
    sections, section_style = detect_sections(records)
    estimates = estimate_chunks(records, sections)
    analysis_text = format_analysis(records, profile, flags,
                                    sections, section_style, estimates)
    print("\n" + analysis_text)

    print(f"\nAsking {RECOMMEND_MODEL} for a chunking recommendation...")
    client = get_client()
    recommendation, usage, prompt, raw_response = ask_mistral(
        client, profile, flags, sections, estimates)

    recommendation_text = format_recommendation(recommendation, usage)
    print("\n" + recommendation_text)

    config = build_config(recommendation, section_style, flags)
    config_text = json.dumps(config, indent=2)
    bar = "─" * 64
    print(f"\n{bar}\nPROPOSED config.json\n{bar}")
    print(config_text)

    # Optional trace file: the full analysis, the exact prompt sent to Mistral,
    # and the raw response — so the recommendation can be inspected without
    # re-running (and without spending another API call).
    if args.trace:
        trace = "\n\n".join([
            analysis_text,
            f"{bar}\nPROMPT SENT TO MISTRAL ({RECOMMEND_MODEL})\n{bar}\n{prompt}",
            f"{bar}\nRAW RESPONSE FROM MISTRAL\n{bar}\n{raw_response}",
            recommendation_text,
            f"{bar}\nPROPOSED config.json\n{bar}\n{config_text}",
        ])
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace, encoding="utf-8")
        print(f"\nTrace written to {trace_path}")

    if args.yes:
        accepted = True
    else:
        try:
            answer = input("\nAccept this config and write config.json? (y/n): ")
            accepted = answer.strip().lower() == "y"
        except EOFError:
            # No interactive terminal (e.g. run non-interactively). Do NOT
            # write — the human-in-the-loop approval is the point of this step.
            print("\n(no interactive input — config.json NOT written; re-run "
                  "with --yes, or answer the prompt in a terminal)")
            return

    if accepted:
        Path(CONFIG_PATH).write_text(json.dumps(config, indent=2),
                                     encoding="utf-8")
        print(f"Wrote {CONFIG_PATH}")
    else:
        print("Not accepted — config.json not written.")


if __name__ == "__main__":
    main()
