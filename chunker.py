"""
chunker.py — turn a .docx into chunks, per config.json  [Phase 3]
=================================================================
Shared by review_chunks.py (preview) and ingest.py (embed + store), so the
chunks previewed are byte-for-byte the chunks ingested.

Two strategies (spec Decision 2):
  A   — one chunk per role / content section (full context per chunk)
  A2  — one chunk per bullet, each carrying a context prefix

ARCHITECTURAL DECISION: semantic chunking, not fixed-size.
Each chunk is a complete structural unit (a role, or a bullet). There is no
chunk-size or overlap parameter — chunk size is an observed *outcome*, reviewed
in review_chunks.py, not an input. See LEARNING_NOTES.md, "Phase 3".

DECISION (Phase 3 review): consecutive job titles collapse into one role.
The CV lists 'Director' then 'Associate Director' back-to-back under one
company — a natural progression, not two roles. The senior/first title is
kept; the dropped title still contributes its dates, so the role shows the
full span (e.g. Dec 2019 – Apr 2022).
"""

import json
import re
from pathlib import Path

from loaders import load_docx

CONFIG_PATH = "config.json"


def load_config(path=CONFIG_PATH):
    """Read config.json (written by analyse.py)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found — run analyse.py first to create it.")
    return json.loads(p.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# DECODE — assign each paragraph a structural role, per config.json
# ──────────────────────────────────────────────
#
# ARCHITECTURAL DECISION (Phase 2): the decode rules live in config.json,
# not in this file. Phase 1 derived roles here from heuristics computed at
# runtime (modal body size, top heading style). Those heuristics were never
# *declared*, *inspectable*, or *per-document* — analyse.py profiled the
# document but chunker.py ignored most of that and re-reasoned its own logic.
# Phase 2 closes that loop: analyse.py + a human author an ordered list of
# fingerprint_rules, chunker.py only executes it. Same code, any document —
# only config.json changes. See SPEC_PHASE2.md, Decision 1.

# The four signal shapes config.json may use. The decode is a pure lookup
# against this grammar — see parse_signal below for why it is not eval().
SUPPORTED_SIGNALS = ("has_numPr", "is_bold", "rendered_size=={n}",
                     "style=={name}")


def parse_signal(signal, paragraph):
    """Evaluate one config signal string against a Paragraph. Returns bool.

    ARCHITECTURAL DECISION: an explicit signal grammar, never eval().
    config.json is a user-editable file. eval() on its contents would execute
    arbitrary Python — a config author would silently gain code execution.
    So we match four fixed signal shapes by hand. The grammar is deliberately
    tiny: a document needing a signal not listed here is a deliberate decision
    to extend SUPPORTED_SIGNALS, reviewed in code — not a config surprise.

      has_numPr            -> the paragraph is a list item
      is_bold              -> the paragraph renders bold
      rendered_size=={n}   -> rendered font size equals n points
      style=={name}        -> the paragraph style name equals {name}

    An unrecognised signal raises ValueError rather than returning False — a
    typo in config.json must fail loudly, not silently skip a rule.
    """
    signal = signal.strip()
    # Reject compound signals up front. Without this check, a signal like
    # `style==Heading 3 && rendered_size==14` would slip past the `style==`
    # branch as a literal style name that never matches anything — a rule that
    # silently does nothing is the exact failure mode Phase 2 exists to fix.
    # The grammar is a decision LIST, not a boolean expression tree: use
    # ordered single-signal rules and let rule order do the job a `&&` would.
    for op in ("&&", "||"):
        if op in signal:
            raise ValueError(
                f"Compound signal not supported: {signal!r}. The grammar is a "
                f"decision list — use ORDERED single-signal rules and let rule "
                f"order disambiguate overlapping cases. Supported signals: "
                f"{', '.join(SUPPORTED_SIGNALS)}.")
    if signal == "has_numPr":
        return paragraph.has_num_pr
    if signal == "is_bold":
        return paragraph.is_bold
    if signal.startswith("rendered_size=="):
        target = signal[len("rendered_size=="):].strip()
        return (paragraph.rendered_size is not None
                and paragraph.rendered_size == float(target))
    if signal.startswith("style=="):
        name = signal[len("style=="):].strip()
        return paragraph.style_name == name
    raise ValueError(
        f"Unknown signal {signal!r} in config.json fingerprint_rules. "
        f"Supported signals: {', '.join(SUPPORTED_SIGNALS)}.")


def decode_role(paragraph, rules):
    """Map one paragraph to a structural role using the ordered config rules.

    Rules are applied IN CONFIG ORDER; the first whose signal matches wins.
    The order is load-bearing (spec Decision 1): a company header may be BOTH
    14pt AND styled 'Heading 3', so a 'rendered_size==14 -> company' rule must
    sit BEFORE 'style==Heading 3 -> job_title'. chunker.py trusts the order
    analyse.py + a human set in config.json — it does not re-reason it.

    No rule matches -> 'body_text' (spec Step 3).
    """
    for rule in rules:
        if parse_signal(rule["signal"], paragraph):
            return rule["role"]
    return "body_text"


def date_span(dates):
    """Collapse a list of date strings into one span.

    The CV lists dates most-recent-first, so the span runs from the start of
    the LAST entry to the end of the FIRST. A single date is returned as-is;
    an empty list yields ''.
    """
    dates = [d for d in dates if d]
    if not dates:
        return ""
    if len(dates) == 1:
        return dates[0]
    def ends(s):  # split a "start – end" string on en/em dash or hyphen
        return [part.strip() for part in re.split(r"\s*[–—-]\s*", s)]
    start = ends(dates[-1])[0]   # earliest entry, its start
    end = ends(dates[0])[-1]     # latest entry, its end
    return f"{start} – {end}"


# ──────────────────────────────────────────────
# GROUP — walk the decoded paragraphs into units
# ──────────────────────────────────────────────

def build_units(records, rules):
    """Walk the decoded paragraphs into structural units.

    decode_role (config-driven) labels each paragraph; this function GROUPS
    those labels into units. Decode is config-driven, grouping is not — the
    config maps a paragraph to a role, the code knows what each role does to
    the running structure (a 'company' opens a role unit, a 'section_header'
    resets the section, etc.).

    A new unit opens at each section header, each company header, and each job
    title that begins a fresh role. CONSECUTIVE job titles collapse into one
    role — the senior/first title is kept, the rest only contribute their
    dates (Phase 3 review decision). This single grouping feeds BOTH
    strategies, so A and A2 are always two views of the same structural read.
    """
    units = []
    section = None
    company = None
    cur = None
    prev_role = None

    def open_unit(kind, label):
        nonlocal cur
        cur = {"kind": kind, "section": section, "company": company,
               "label": label, "titles": [], "dates": [],
               "bullets": [], "body": []}
        units.append(cur)

    for rec in records:
        role = decode_role(rec, rules)
        text, date = rec.text, rec.date

        if role == "section_header":
            section = text
            company = None
            open_unit("section", text)
        elif role == "company":
            company = text
            open_unit("role", text)
        elif role == "job_title":
            if prev_role == "job_title" and cur is not None:
                # consecutive title — collapse: keep only its date, drop text
                if date:
                    cur["dates"].append(date)
            else:
                # a title that begins a fresh role (new role, same company)
                if cur is None or cur["titles"] or cur["bullets"] or cur["body"]:
                    open_unit("role", text)
                cur["titles"].append(text)
                if date:
                    cur["dates"].append(date)
        elif role == "bullet":
            if cur is None:
                open_unit("section", section or "(document)")
            cur["bullets"].append(text)
        else:  # body_text (the decode_role default for an unmatched paragraph)
            if cur is None:
                open_unit("section", section or "(document)")
            cur["body"].append(text)
            if date:                       # body lines can carry a date too
                cur["dates"].append(date)

        prev_role = role

    return units


# ──────────────────────────────────────────────
# ASSEMBLE — units into chunks, per strategy
# ──────────────────────────────────────────────

class _SafeDict(dict):
    """Yields '' for any missing key, so a prefix template can reference a
    field this document does not have without raising KeyError."""
    def __missing__(self, key):
        return ""


def _fields(unit):
    """Semantic fields available to a prefix template and to metadata.

    `company` is set ONLY for a genuine job entry — a unit that has a job title
    or bullets. Size-prominent headings that are NOT jobs (the person's name,
    'Core Skills', other CV sub-headings) are size-14 'headers' too, but they
    are not companies. Their text is kept as `heading` so it still anchors the
    chunk text, without polluting the company metadata used for filtering.
    """
    title = " / ".join(unit["titles"])
    heading = unit["company"] or ""
    is_job = bool(unit["titles"] or unit["bullets"])
    return {
        "heading": heading,
        "company": heading if is_job else "",
        "job_title": title,
        "title": title,
        "section_name": unit["section"] or "",
        "section": unit["section"] or "",
        "dates": date_span(unit["dates"]),
    }


def _render_prefix(template, fields):
    """Fill the prefix template, then drop segments left empty.

    A role with no job title (e.g. Imagination) would otherwise render an
    empty '|  |' gap. We collapse those so the prefix stays clean.
    """
    if not template:
        return ""
    s = template.format_map(_SafeDict(fields))
    while re.search(r"\|\s*\|", s):        # empty middle segment(s)
        s = re.sub(r"\|\s*\|", "|", s)
    s = re.sub(r"\[\s*\|\s*", "[", s)      # empty leading segment
    s = re.sub(r"\s*\|\s*\]", "]", s)      # empty trailing segment
    return re.sub(r"\s{2,}", " ", s).strip()


def _metadata(unit, strategy, index, text, is_bullet):
    """Flat, ChromaDB-safe metadata — str / int / bool only (no None, no lists)."""
    f = _fields(unit)
    return {
        "company": f["company"],
        "job_title": f["job_title"],
        "section_name": f["section_name"],
        "dates": f["dates"],
        "strategy": strategy,
        "is_bullet": is_bullet,
        "chunk_index": index,
        "word_count": len(text.split()),
    }


def _chunk(strategy, index, text, unit, is_bullet):
    return {
        "id": f"cv_{strategy.lower()}_{index:03d}",
        "text": text,
        "metadata": _metadata(unit, strategy, index, text, is_bullet),
    }


def chunks_strategy_a(units):
    """Strategy A — one chunk per unit (a role, or a content section)."""
    chunks = []
    for unit in units:
        if not unit["body"] and not unit["bullets"]:
            continue  # an empty section heading with no content — skip
        f = _fields(unit)
        # Header line: the unit's heading / title / dates for a role; section /
        # dates otherwise. Uses `heading` (raw prominent text), not `company`,
        # so a non-job heading like 'Core Skills' still anchors its chunk.
        if f["heading"] or f["job_title"]:
            head = [b for b in (f["heading"], f["job_title"], f["dates"]) if b]
        else:
            head = [b for b in (f["section_name"], f["dates"]) if b]
        parts = []
        if head:
            parts.append(" — ".join(head))
        parts.extend(unit["body"])
        parts.extend(unit["bullets"])
        text = "\n".join(p for p in parts if p).strip()
        chunks.append(_chunk("A", len(chunks), text, unit, is_bullet=False))
    return chunks


def chunks_strategy_a2(units, prefix_template):
    """Strategy A2 — one chunk per bullet (+ one per non-bullet section).

    Each bullet carries a context prefix so it stays self-contained — the
    prefix does the job that overlap does in fixed-size chunking (spec
    Decision 2: the prefix is load-bearing for retrieval, not decoration).
    """
    chunks = []
    for unit in units:
        if unit["bullets"]:
            prefix = _render_prefix(prefix_template, _fields(unit))
            for bullet in unit["bullets"]:
                text = f"{prefix} {bullet}".strip()
                chunks.append(_chunk("A2", len(chunks), text, unit,
                                     is_bullet=True))
            body = " ".join(unit["body"]).strip()
            if body:  # non-bullet body inside a role unit — one extra chunk
                chunks.append(_chunk("A2", len(chunks),
                                     f"{prefix} {body}".strip(), unit,
                                     is_bullet=False))
        else:
            # a content section with no bullets — one chunk for its content,
            # with a light [Section] prefix (the role-oriented template would
            # be mostly empty placeholders here).
            parts = []
            if unit["titles"]:
                parts.append(" / ".join(unit["titles"]))
            parts.extend(unit["body"])
            body = "\n".join(p for p in parts if p).strip()
            if body:
                tag = f"[{unit['section']}] " if unit["section"] else ""
                chunks.append(_chunk("A2", len(chunks), f"{tag}{body}".strip(),
                                     unit, is_bullet=False))
    return chunks


def all_chunks(docx_path, config=None):
    """Chunk a document under BOTH strategies. Returns {'A': [...], 'A2': [...]}.

    One document read feeds both — A and A2 are two views of the same units.
    """
    config = config or load_config()
    rules = config.get("fingerprint_rules")
    if not rules:
        raise ValueError(
            "config.json has no 'fingerprint_rules' — run analyse.py to "
            "generate them. A config-driven chunker cannot decode without "
            "rules, and silently producing whole-document mega-chunks is the "
            "exact failure mode Phase 2 exists to fix. Refusing to chunk.")
    records = load_docx(docx_path)
    units = build_units(records, rules)
    return {
        "A": chunks_strategy_a(units),
        "A2": chunks_strategy_a2(units, config.get("prefix_template", "")),
    }
