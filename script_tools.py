"""
script_tools.py — deterministic (no-LLM) helpers for ScriptCraft.

Everything here is pure Python: cheap, instant, and unit-testable.
  * build_registry / mentioned_entities : find characters & places in the Show Bible
  * repair_script   : fix *objective* defects (fences, blank lines inside dialogue,
                      lowercase headings, chatty preamble, runaway loops, cut-off tails)
  * validate_script : rule-based checks -> {"errors": [...], "warns": [...]}
  * parse_json_loose: tolerant JSON parsing for LLM critic output
"""
import re
import json
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

# ----------------------------------------------------------------------------
# Regexes / small helpers
# ----------------------------------------------------------------------------
HEADING_RE = re.compile(r"^\s*(INT\.?/EXT\.?|EXT\.?/INT\.?|INT\.|EXT\.|EST\.|I/E\.?)\s", re.I)
TRANSITION_RE = re.compile(r"^\s*(FADE (IN|OUT)|CUT TO|SMASH CUT TO|MATCH CUT TO|DISSOLVE TO|BACK TO)\b|TO:\s*$", re.I)
CHATTY_OPEN_RE = re.compile(r"^\s*(sure|certainly|here is|here's|here are|of course|okay|ok,|absolutely)\b", re.I)
CHATTY_CLOSE_RE = re.compile(r"^\s*(let me know|i hope|feel free|would you like|note:|\(note|---+$|\*\*\*+$)", re.I)
COLON_DIALOGUE_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,25}:\s+\S")

STOP_TOKENS = {"THE", "AND", "OF", "FOR", "WITH", "IN", "AT", "ON", "A", "AN"}
NOT_ENTITIES = {"CHARACTERS", "CHARACTER", "SETTINGS", "SETTING", "LOCATIONS", "LORE", "SHOW BIBLE",
                "WORLD", "OVERVIEW", "PLOT", "THEMES", "NOTES", "RULES", "TIMELINE"}


def _words(text: str) -> Set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        out.add(w)
        out.add(w.split("'")[0])
    return out


def _tokens(name: str) -> Set[str]:
    toks = {t.strip("'") for t in re.findall(r"[A-Z0-9']+", name.upper())}
    return {t for t in toks if len(t) >= 3 and t not in STOP_TOKENS}


def is_transition(s: str) -> bool:
    return bool(TRANSITION_RE.search(s))


def _strip_ext(s: str) -> str:
    return re.sub(r"\(.*?\)", "", s).strip()


def cue_like(line: str) -> bool:
    """ALL-CAPS, short, not a heading/transition, no sentence punctuation."""
    s = _strip_ext(line.strip())
    if not s or s != s.upper() or len(re.findall(r"[A-Z]", s)) < 2:
        return False
    if len(s.split()) > 4 or len(s) > 30:
        return False
    if HEADING_RE.match(s) or is_transition(s) or re.search(r"[.!?,;:]$", s):
        return False
    return True


def _known(cue: str, names: Set[str]) -> bool:
    c = _strip_ext(cue).upper()
    return c in names or (bool(c.split()) and all(t in names for t in c.split()))


def is_parenthetical(s: str) -> bool:
    s = s.strip()
    return s.startswith("(") and s.endswith(")")


def extract_cues(text: str) -> List[str]:
    """Character cues that start a proper dialogue block (blank line before, text right after)."""
    lines = text.splitlines()
    cues = []
    for i, l in enumerate(lines[:-1]):
        prev_blank = (i == 0) or not lines[i - 1].strip()
        if prev_blank and cue_like(l) and lines[i + 1].strip():
            cues.append(_strip_ext(l.strip()))
    return cues


# ----------------------------------------------------------------------------
# Show-bible entity registry
# ----------------------------------------------------------------------------
def build_registry(texts_by_file: Dict[str, str]) -> Dict[str, dict]:
    """Find ALL-CAPS headings / 'NAME:' entries in the bible. Files named *character* -> kind='character'."""
    reg: Dict[str, dict] = {}
    for fname, text in texts_by_file.items():
        stem = fname.rsplit(".", 1)[0].upper()
        kind = "character" if "character" in fname.lower() else "other"
        for raw in text.splitlines():
            line = re.sub(r"^[#>\s*_\-]+", "", raw)
            line = re.sub(r"[*_\s]+$", "", line).strip()
            name = None
            m = re.match(r"^([A-Z][A-Z0-9 .'\-]{2,40}?)\s*:", line)
            if m:
                name = m.group(1).strip()
            elif (line and line == line.upper() and re.search(r"[A-Z]{3}", line)
                  and len(line.split()) <= 6 and not re.search(r"[.!?,;:]$", line)):
                name = line
            if not name or name in NOT_ENTITIES or name == stem or name in reg:
                continue
            toks = _tokens(name)
            if toks:
                reg[name] = {"kind": kind, "file": fname, "tokens": toks}
    return reg


def mentioned_entities(prompt: str, registry: Dict[str, dict], kinds=("character", "other")) -> List[str]:
    words = _words(prompt)
    hits = []
    for name, e in registry.items():
        if e["kind"] not in kinds:
            continue
        toks = {t.lower() for t in e["tokens"]}
        n = len(toks & words)
        need = 1 if e["kind"] == "character" else min(2, len(toks))
        if n >= need and n > 0:
            hits.append(name)
    return hits


def name_present(text: str, name: str, registry: Dict[str, dict]) -> bool:
    toks = {t.lower() for t in registry[name]["tokens"]}
    return bool(toks & _words(text))


# ----------------------------------------------------------------------------
# Repair (objective defects only)
# ----------------------------------------------------------------------------
def _join_dialogue(lines: List[str], names: Set[str]) -> Tuple[List[str], int]:
    """Remove blank lines between CUE / (parenthetical) / dialogue — they break a Fountain dialogue block."""
    out, i, n, joined = [], 0, len(lines), 0

    def next_content(k):
        while k < n and not lines[k].strip():
            k += 1
        return k

    while i < n:
        s = lines[i].strip()
        prev_blank = (not out) or (not out[-1].strip())
        if prev_blank and s and cue_like(s) and _known(s, names):
            out.append(s)
            i += 1
            while True:
                k = next_content(i)
                if k >= n:
                    break
                nl = lines[k].strip()
                if k > i and not HEADING_RE.match(nl) and not is_transition(nl) and not (cue_like(nl) and _known(nl, names)):
                    joined += 1
                    i = k
                if i < n and is_parenthetical(lines[i]):
                    out.append(lines[i].strip())
                    i += 1
                    continue
                break
            continue
        out.append(lines[i])
        i += 1
    return out, joined


def _fix_mixed_case_cues(lines: List[str], names: Set[str]) -> Tuple[List[str], int]:
    fixed, out = 0, list(lines)
    for i, l in enumerate(out):
        s = l.strip()
        prev_blank = (i == 0) or not out[i - 1].strip()
        nxt = next((x.strip() for x in out[i + 1:] if x.strip()), "")
        if (prev_blank and nxt and s and s != s.upper() and len(s.split()) <= 3
                and re.fullmatch(r"[A-Za-z .'\-]+", s) and not re.search(r"[.!?,;:]$", s)
                and s.upper() in names):
            out[i] = s.upper()
            fixed += 1
    return out, fixed


def _dialogue_lines(block: str) -> List[str]:
    res = []
    for l in block.splitlines():
        s = l.strip()
        if not s or cue_like(s) or is_parenthetical(s) or HEADING_RE.match(s):
            continue
        norm = re.sub(r"[^a-z0-9 ]", "", s.lower())
        if len(norm.split()) >= 4:
            res.append(norm)
    return res


def trim_loops(text: str) -> Tuple[str, bool]:
    """Detect runaway repetition; cut the script just before the first repeat."""
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    keys = [re.sub(r"\s+", " ", b.strip().lower()) for b in blocks]
    block_counts = Counter(keys)
    line_counts = Counter(l for b in blocks for l in _dialogue_lines(b))
    bad_blocks = {k for k, c in block_counts.items() if c >= 3}
    bad_lines = {l for l, c in line_counts.items() if c >= 3}
    if not bad_blocks and not bad_lines:
        return text.strip(), False
    seen_b, seen_l, kept = set(), set(), []
    for b, k in zip(blocks, keys):
        dl = _dialogue_lines(b)
        if (k in bad_blocks and k in seen_b) or any(l in bad_lines and l in seen_l for l in dl):
            break
        seen_b.add(k)
        seen_l.update(dl)
        kept.append(b)
    return "\n\n".join(kept), True


def repair_script(text: str, known_names: Iterable[str] = (), truncated: bool = False) -> Tuple[str, List[str], bool]:
    """Returns (repaired_text, list_of_repairs_applied, loop_found)."""
    repairs: List[str] = []
    names = {n.upper() for n in known_names}
    lines = [l.rstrip() for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    # 1. code fences / markdown headers
    cleaned = []
    for l in lines:
        if l.strip().startswith("```"):
            repairs.append("removed code fence")
            continue
        m = re.match(r"^\s*#{1,6}\s+(.*)$", l)
        if m:
            repairs.append("removed markdown header")
            l = m.group(1)
        cleaned.append(l)
    lines = cleaned

    # 2. chatty preamble before the first scene heading
    first_h = next((i for i, l in enumerate(lines) if HEADING_RE.match(l)), None)
    if first_h:
        pre = [l for l in lines[:first_h] if l.strip()]
        if pre and len(pre) <= 3 and any(re.search(r"[a-z]", l) for l in pre) and not any(is_transition(l) for l in pre):
            lines = lines[first_h:]
            repairs.append("dropped chatty preamble")

    # 3. chatty epilogue
    while True:
        idx = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
        if idx is not None and CHATTY_CLOSE_RE.match(lines[idx]):
            lines = lines[:idx]
            repairs.append("dropped chatty epilogue")
        else:
            break

    # 4. scene headings -> ALL CAPS
    for i, l in enumerate(lines):
        if HEADING_RE.match(l) and l != l.upper():
            lines[i] = l.upper()
            repairs.append("uppercased scene heading")

    # 5. mixed-case cues, blank lines inside dialogue blocks
    lines, nfix = _fix_mixed_case_cues(lines, names)
    if nfix:
        repairs.append(f"uppercased {nfix} character cue(s)")
    # also trust cues that already appear properly in this very draft
    names_plus = names | {c.upper() for c in extract_cues("\n".join(lines))}
    lines, njoin = _join_dialogue(lines, names_plus)
    if njoin:
        repairs.append(f"removed illegal blank lines inside {njoin} dialogue block(s)")

    # 6. collapse blank runs
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    # 7. runaway loops
    out, loop = trim_loops(out)
    if loop:
        repairs.append("trimmed repetitive loop")

    # 8. cut-off tail
    if truncated:
        blocks = re.split(r"\n\s*\n", out)
        if len(blocks) > 1:
            out = "\n\n".join(blocks[:-1])
            repairs.append("dropped unfinished final block")

    return out, repairs, loop


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
GENERIC_CUES = {"VOICE", "ANNOUNCER", "NARRATOR", "RADIO", "TV", "MAN", "WOMAN", "BOY", "GIRL", "CROWD",
                "BARISTA", "CUSTOMER", "WAITER", "WAITRESS", "GUARD", "OFFICER", "DOCTOR", "NURSE", "ALL", "BOTH"}


def count_broken_blocks(text: str, names: Set[str]) -> int:
    """Known CUE lines followed by a blank line (illegal in Fountain: it breaks the dialogue block)."""
    lines, n = text.splitlines(), 0
    for i, l in enumerate(lines[:-1]):
        prev_blank = (i == 0) or not lines[i - 1].strip()
        if prev_blank and l.strip() and cue_like(l) and _known(l, names) and not lines[i + 1].strip():
            nxt = next((x.strip() for x in lines[i + 1:] if x.strip()), "")
            if nxt and not HEADING_RE.match(nxt) and not is_transition(nxt):
                n += 1
    return n


def validate_script(text: str, *, user_prompt: str = "", registry: Dict[str, dict] = None,
                    min_words: int = 150, max_words: int = 900,
                    truncated: bool = False, loop_found: bool = False) -> dict:
    registry = registry or {}
    errors: List[str] = []
    warns: List[str] = []
    t = text.strip()
    words = len(t.split())
    stats = {"words": words}

    if not t:
        return {"errors": ["empty"], "warns": [], "stats": stats}

    first = next((l for l in t.splitlines() if l.strip()), "")
    heads = [l for l in t.splitlines() if HEADING_RE.match(l)]
    if not heads:
        errors.append("no_scene_heading")
    elif not (HEADING_RE.match(first) or is_transition(first)):
        warns.append("does_not_start_with_heading")
    if any(h != h.upper() for h in heads):
        errors.append("heading_not_caps")

    if "```" in t or re.search(r"^\s*#{1,6}\s", t, re.M) or re.search(r"^\s*\|.+\|\s*$", t, re.M):
        errors.append("markdown")
    if CHATTY_OPEN_RE.match(first):
        errors.append("chatty_opener")

    cues = extract_cues(t)
    stats["dialogue_blocks"] = len(cues)
    if len(cues) < 2:
        errors.append("no_dialogue")
    if any(COLON_DIALOGUE_RE.match(l) and not is_transition(l) for l in t.splitlines()):
        errors.append("colon_dialogue")

    if words < min_words:
        errors.append("too_short")
    if words > max_words:
        errors.append("too_long")
    if loop_found:
        errors.append("loop_detected")
    if truncated:
        errors.append("truncated")

    # characters named in the director's request must appear in the scene
    for name in mentioned_entities(user_prompt, registry, kinds=("character",)):
        if not name_present(t, name, registry):
            errors.append(f"missing_character:{name}")

    # speaking characters unknown to the bible (soft: new minor characters are allowed)
    if registry:
        known = set()
        for n, e in registry.items():
            known.add(n.upper())
            known |= e["tokens"]
        known |= {w.upper() for w in _words(user_prompt)}
        unknown = sorted({c for c in cues if not _known(c, known) and c.upper() not in GENERIC_CUES})
        if unknown:
            warns.append("unknown_speakers:" + ", ".join(unknown))

    if registry:
        allnames = set()
        for n_, e_ in registry.items():
            allnames.add(n_.upper()); allnames |= e_["tokens"]
        broken = count_broken_blocks(t, allnames | {c.upper() for c in cues})
        if broken:
            warns.append(f"broken_dialogue_blocks:{broken}")

    if re.search(r"(?i)\b([\w' ]{6,40}[.!?])\s+\1", t):
        warns.append("stuttered_repeat")

    return {"errors": errors, "warns": warns, "stats": stats}


ERROR_HELP = {
    "empty": "The output was empty. Write the full scene.",
    "no_scene_heading": "Start with a scene heading in ALL CAPS, e.g. INT. LOCATION - TIME.",
    "heading_not_caps": "Scene headings must be ALL CAPS.",
    "markdown": "Output raw screenplay text only: no code fences, '#' headers or tables.",
    "chatty_opener": "Do not start with conversational filler; begin directly with the scene heading.",
    "no_dialogue": "Include real dialogue: at least two dialogue blocks (CHARACTER NAME on its own line, then the lines).",
    "colon_dialogue": "Do not write dialogue as 'NAME: line'. Put the NAME in ALL CAPS on its own line and the speech on the next line.",
    "too_short": "The scene was too short. Write a complete scene with a beginning, a turn, and a closing beat.",
    "too_long": "The scene was too long. Tighten it to a single focused scene and end it cleanly.",
    "loop_detected": "You fell into a repetitive loop. NEVER repeat a line of dialogue; every exchange must move the scene forward and the scene must reach a clear ending.",
    "truncated": "The output was cut off. Write a tighter scene (about 350-450 words) that finishes cleanly.",
}


def explain_error(code: str) -> str:
    if code.startswith("missing_character:"):
        return f"The director's request names {code.split(':', 1)[1]}, but they never appear in the scene. Include them."
    return ERROR_HELP.get(code, code)


# ----------------------------------------------------------------------------
# JSON helper for critic output
# ----------------------------------------------------------------------------
def parse_json_loose(s: str):
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def grounded(evidence: str, *sources: str, threshold: float = 0.6) -> bool:
    """Is the critic's quoted evidence actually present (loosely) in the context/request?"""
    ev = {w for w in re.findall(r"[a-z0-9']+", (evidence or "").lower()) if len(w) >= 3}
    if not ev:
        return False
    src = set()
    for s in sources:
        src |= set(re.findall(r"[a-z0-9']+", (s or "").lower()))
    return len(ev & src) / len(ev) >= threshold