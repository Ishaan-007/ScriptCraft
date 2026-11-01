"""
rag_engine.py — ScriptCraft agentic screenplay pipeline (LangGraph + Groq).

Graph:
  retrieve ─▶ draft ─▶ validate ─▶ critic ─▶ finalize
                │  ▲       │  ▲        │
                │  │       ▼  │        ▼
                │  └─ prepare_revision ◀┘        (max N revisions)
                ├─▶ rewrite_query ─▶ retrieve    (INSUFFICIENT_CONTEXT, once — same as before)
                └─▶ fallback                      (INSUFFICIENT_CONTEXT twice)

Safety nets so quality can never fall below the old single-shot pipeline:
  * every draft is kept as a candidate; the BEST one is returned (fewest rule errors, then best critic score,
    ties -> earliest draft), so a bad revision can never replace a better earlier draft;
  * deterministic repair only fixes objective defects (fences, blank lines inside dialogue, loops, cut-off tails);
  * the critic is fail-open (API/parse errors => skipped) and its objections only count when grounded in the context;
  * ask_agent(..., agentic=False) reproduces the legacy single-shot behaviour for A/B comparison.
"""
import os
import re
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional

from groq import Groq, APIError, BadRequestError
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

from script_tools import (
    build_registry, mentioned_entities, repair_script, validate_script,
    explain_error, parse_json_loose, grounded,
)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class AgentConfig:
    # Models can be swapped without code changes:  export SCRIPTCRAFT_CRITIC_MODEL=...
    drafter_model: str = field(default_factory=lambda: os.getenv("SCRIPTCRAFT_DRAFT_MODEL", "openai/gpt-oss-20b"))
    critic_model: str = field(default_factory=lambda: os.getenv("SCRIPTCRAFT_CRITIC_MODEL", "openai/gpt-oss-120b"))
    critic_fallback_model: str = field(default_factory=lambda: os.getenv("SCRIPTCRAFT_CRITIC_FALLBACK", "openai/gpt-oss-20b"))
    temperature: float = 0.7
    draft_max_tokens: int = 3000          # hard stop against runaway generations (reasoning tokens included)
    critic_max_tokens: int = 1500
    drafter_reasoning_effort: Optional[str] = None   # None = model default (same as legacy behaviour)
    critic_reasoning_effort: str = "medium"
    max_revisions: int = 2                # total extra drafts after the first one
    min_words: int = 150
    max_words: int = 900
    target_words: tuple = (300, 500)
    entity_boost: bool = True             # also pull bible chunks for characters/places named in the request
    max_boost_chunks: int = 3
    use_critic: bool = True
    pass_default_score: int = 7           # rank value when the critic was unavailable


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
class ScriptState(TypedDict, total=False):
    user_prompt: str
    search_query: str
    k_chunks: int
    context: str
    sources: List[str]
    agentic: bool
    use_critic: bool
    # drafting
    draft: str
    finish_reason: str
    draft_failed: bool
    attempt: int
    raw_first_draft: str
    revision_notes: str
    revise_mode: str          # "regenerate" | "revise"
    revise_base: str
    revisions_used: int
    retry_count: int          # number of rejected drafts / retrievals (shown in the UI)
    # checking
    candidates: list
    validation: dict
    critique: dict
    final_script: str
    trace: Annotated[list, operator.add]


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
LEGACY_RULES = """RULES:
- If the context does not contain enough information to write the requested scene accurately, output exactly: INSUFFICIENT_CONTEXT
- Otherwise, output STRICTLY in standard Fountain/Screenplay format.
- Scene Headings must be ALL CAPS (e.g., INT. COFFEE SHOP - DAY).
- Character names above dialogue must be ALL CAPS and centered (or on their own line).
- Do NOT include conversational filler, markdown formatting blocks like ``` or citations in the text. Just the raw script."""

AGENTIC_RULES = """
- Dialogue block layout: CHARACTER NAME, then an optional (parenthetical), then the spoken lines, all on consecutive lines with NO blank lines between them. Put exactly one blank line between blocks.
- Write ONE complete scene of roughly {lo}-{hi} words with a beginning, a turn, and a closing beat, then stop.
- Never repeat a line of dialogue or a stage direction. Every exchange must move the scene forward.
- Keep every fact consistent with the CONTINUITY CONTEXT (handedness/sides, jobs, relationships, names, setting details). Invent only minor details that do not contradict it."""

CRITIC_SYSTEM = """You are a meticulous script supervisor checking a screenplay draft for continuity against a Show Bible.

You receive: the DIRECTOR'S REQUEST, the CONTINUITY CONTEXT (retrieved Show Bible excerpts) and the DRAFT SCENE.
Find only OBJECTIVE problems:
1. continuity - facts in the draft that contradict the CONTEXT (wrong side/limb, job, relationship, name, setting detail, or a character knowing/doing something the context rules out).
2. character - speech or behaviour clearly at odds with the described personality (e.g. a character described as speaking in short sentences giving long speeches).
3. request - the draft fails to do something the DIRECTOR'S REQUEST explicitly asks for (a named character, an event, a requested beat).
4. coherence - lines that make no sense in context, or dialogue that loops.

Rules:
- A violation is "major" ONLY if you can quote the exact CONTEXT line (or request phrase) it conflicts with; put that quote in "evidence".
- Do NOT flag creative additions the context does not forbid (new minor characters, invented dialogue, atmosphere, props).
- Do NOT flag style, taste, pacing, or screenplay formatting.
- If you are unsure, use severity "minor".

Return ONLY a JSON object, no prose:
{"score": <integer 1-10 for overall fidelity and quality>,
 "violations": [{"type": "continuity|character|request|coherence", "severity": "major|minor",
                 "evidence": "<exact quote from CONTEXT or REQUEST>",
                 "problem": "<what the draft says that conflicts>",
                 "fix": "<concrete minimal fix>"}]}
If the draft is faithful return {"score": 9, "violations": []}."""


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class ScriptCraftRAG:
    def __init__(self, groq_api_key: str, config: Optional[AgentConfig] = None, *,
                 client=None, vector_store=None, bible_dir=None):
        self.base_dir = Path(__file__).resolve().parent
        self.bible_dir = Path(bible_dir) if bible_dir else self.base_dir / "show_bible"
        self.cfg = config or AgentConfig()
        self.client = client or Groq(api_key=groq_api_key, max_retries=3, timeout=60.0)
        self.model_name = self.cfg.drafter_model          # kept for backwards compatibility

        # Load & chunk the Show Bible
        docs, texts = [], {}
        for file_path in sorted(self.bible_dir.glob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            texts[file_path.name] = text
            docs.append(Document(page_content=text, metadata={"source": file_path.name}))
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Entity registry (characters / places) used by the validator and the retrieval boost
        self.registry = build_registry(texts)
        self.known_names = set()
        for n, e in self.registry.items():
            self.known_names.add(n.upper())
            self.known_names |= e["tokens"]
        self.entity_chunks = {}
        for name in self.registry:
            pat = re.compile(rf"(?m)^\W*{re.escape(name)}\W*$")
            self.entity_chunks[name] = [c for c in chunks if pat.search(c.page_content)]

        # Vector store (injectable for offline tests)
        if vector_store is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
        self.vector_store = vector_store

        self.graph = self._build_graph()

    # ------------------------------------------------------------------ LLM plumbing
    def _chat(self, model, messages, *, temperature, max_tokens=None, json_mode=False, reasoning_effort=None):
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort and "gpt-oss" in model:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not json_mode:
                raise
            kwargs.pop("response_format")
            resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return (choice.message.content or "").strip(), choice.finish_reason

    def _system_prompt(self, context: str, agentic: bool) -> str:
        rules = LEGACY_RULES
        if agentic:
            lo, hi = self.cfg.target_words
            rules += AGENTIC_RULES.format(lo=lo, hi=hi)
        return f"""You are an expert Hollywood screenwriter.
Write a scene based on the USER PROMPT.
You MUST use the CONTINUITY CONTEXT to ensure character personalities, lore, and settings are accurate.

{rules}

CONTINUITY CONTEXT:
{context}"""

    # ------------------------------------------------------------------ NODES
    def retrieve_context(self, state: ScriptState):
        k = state.get("k_chunks", 3)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        retrieved = list(retriever.invoke(state.get("search_query", state["user_prompt"])))

        boosted = []
        if state.get("agentic", True) and self.cfg.entity_boost:
            have = {d.page_content.strip() for d in retrieved}
            for name in mentioned_entities(state["user_prompt"], self.registry):
                cs = self.entity_chunks.get(name) or []
                if cs and not any(c.page_content.strip() in have for c in cs):
                    boosted.append(cs[0])
                    have.add(cs[0].page_content.strip())
                if len(boosted) >= self.cfg.max_boost_chunks:
                    break

        parts, sources = [], []
        for d in retrieved + boosted:
            src = d.metadata.get("source", "unknown file")
            if src not in sources:
                sources.append(src)
            parts.append(f"### Source: {src}\n{d.page_content.strip()}")
        ev = {"step": "retrieve", "k": k, "chunks": len(retrieved), "boosted": len(boosted)}
        return {"context": "\n\n".join(parts), "sources": sources, "trace": [ev]}

    def draft_scene(self, state: ScriptState):
        agentic = state.get("agentic", True)
        attempt = state.get("attempt", 0)
        system_prompt = self._system_prompt(state.get("context", ""), agentic)

        notes = state.get("revision_notes", "")
        if notes and state.get("revise_mode") == "revise":
            user_msg = (f"{state['user_prompt']}\n\n---\nYOUR PREVIOUS DRAFT:\n<<<\n{state.get('revise_base', '')}\n>>>\n\n"
                        f"A script supervisor found these problems:\n{notes}\n\n"
                        "Rewrite the COMPLETE scene fixing only these problems. Keep everything else (voice, good lines, "
                        "structure) as it is. Output only the raw screenplay.")
        elif notes:
            user_msg = (f"{state['user_prompt']}\n\n---\nYour previous attempt was rejected:\n{notes}\n"
                        "Write the complete scene again and avoid these problems. Output only the raw screenplay.")
        else:
            user_msg = state["user_prompt"]

        try:
            raw, finish = self._chat(
                self.cfg.drafter_model,
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.draft_max_tokens if agentic else None,
                reasoning_effort=self.cfg.drafter_reasoning_effort if agentic else None,
            )
        except APIError as e:
            if attempt == 0:
                raise                                  # same behaviour as before for the first draft
            return {"draft": "", "draft_failed": True, "revision_notes": "",
                    "trace": [{"step": "draft", "attempt": attempt, "error": type(e).__name__}]}

        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        out = {"draft": raw, "finish_reason": finish, "attempt": attempt + 1, "draft_failed": False,
               "revision_notes": "",
               "trace": [{"step": "draft", "attempt": attempt + 1, "model": self.cfg.drafter_model,
                          "mode": state.get("revise_mode", "initial") if notes else "initial",
                          "words": len(raw.split()), "finish": finish}]}
        if attempt == 0:
            out["raw_first_draft"] = raw
        return out

    def route_after_draft(self, state: ScriptState):
        if state.get("draft_failed"):
            return "finalize"
        if "INSUFFICIENT_CONTEXT" in state.get("draft", "").upper():
            return "retry" if state.get("retry_count", 0) < 1 else "fallback"
        if not state.get("agentic", True):
            return "finalize"
        return "validate"

    def validate_draft(self, state: ScriptState):
        truncated = state.get("finish_reason") == "length"
        text, repairs, loop = repair_script(state["draft"], self.known_names, truncated=truncated)
        report = validate_script(text, user_prompt=state["user_prompt"], registry=self.registry,
                                 min_words=self.cfg.min_words, max_words=self.cfg.max_words,
                                 truncated=truncated, loop_found=loop)
        cand = {"text": text, "attempt": state.get("attempt", 1), "errors": report["errors"],
                "warns": report["warns"], "critic_score": None, "repairs": repairs}
        return {"candidates": list(state.get("candidates", [])) + [cand], "validation": report,
                "trace": [{"step": "validate", "attempt": cand["attempt"], "repairs": repairs,
                           "errors": report["errors"], "warns": report["warns"], "words": report["stats"]["words"]}]}

    def route_after_validate(self, state: ScriptState):
        if state["validation"]["errors"]:
            return "revise" if state.get("revisions_used", 0) < self.cfg.max_revisions else "finalize"
        return "critic" if (self.cfg.use_critic and state.get("use_critic", True)) else "finalize"

    # ---- critic ---------------------------------------------------------
    def critique(self, user_prompt: str, context: str, script: str, models: Optional[List[str]] = None) -> Optional[dict]:
        """LLM continuity check. Returns None if every critic model fails (fail-open)."""
        models = models or [self.cfg.critic_model, self.cfg.critic_fallback_model]
        user = (f"DIRECTOR'S REQUEST:\n{user_prompt}\n\nCONTINUITY CONTEXT:\n{context}\n\nDRAFT SCENE:\n{script}")
        seen = set()
        for m in models:
            if m in seen:
                continue
            seen.add(m)
            try:
                raw, _ = self._chat(m, [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}],
                                    temperature=0.2, max_tokens=self.cfg.critic_max_tokens, json_mode=True,
                                    reasoning_effort=self.cfg.critic_reasoning_effort)
            except APIError:
                continue
            data = parse_json_loose(raw)
            if not isinstance(data, dict):
                continue
            viols = [v for v in (data.get("violations") or []) if isinstance(v, dict)]
            major = [v for v in viols if str(v.get("severity", "")).lower() == "major"
                     and grounded(str(v.get("evidence", "")), context, user_prompt)]
            try:
                score = max(1, min(10, int(round(float(data.get("score", self.cfg.pass_default_score))))))
            except (TypeError, ValueError):
                score = self.cfg.pass_default_score
            return {"score": score, "violations": viols, "major": major, "needs_revision": bool(major), "model": m}
        return None

    def critic_node(self, state: ScriptState):
        cands = list(state["candidates"])
        cur = dict(cands[-1])
        verdict = self.critique(state["user_prompt"], state.get("context", ""), cur["text"])
        if verdict is None:
            return {"critique": {"skipped": True, "needs_revision": False},
                    "trace": [{"step": "critic", "attempt": cur["attempt"], "skipped": True}]}
        cur["critic_score"] = verdict["score"]
        cur["violations"] = verdict["major"]
        cands[-1] = cur
        return {"candidates": cands, "critique": verdict,
                "trace": [{"step": "critic", "attempt": cur["attempt"], "model": verdict["model"], "score": verdict["score"],
                           "major": [{k: v.get(k) for k in ("type", "evidence", "problem", "fix")} for v in verdict["major"]],
                           "minor": len(verdict["violations"]) - len(verdict["major"])}]}

    def route_after_critic(self, state: ScriptState):
        if state.get("critique", {}).get("needs_revision") and state.get("revisions_used", 0) < self.cfg.max_revisions:
            return "revise"
        return "finalize"

    def prepare_revision(self, state: ScriptState):
        errors = state.get("validation", {}).get("errors", [])
        if errors:                                         # rule-based failure -> regenerate with feedback
            notes = "\n".join(f"- {explain_error(e)}" for e in errors)
            mode, base = "regenerate", ""
        else:                                              # critic objection -> targeted revision of the draft
            majors = state["critique"]["major"]
            notes = "\n".join(
                f"- [{v.get('type', 'continuity')}] The context says: \"{v.get('evidence', '')}\" but the draft: "
                f"{v.get('problem', '')}. Fix: {v.get('fix', '')}" for v in majors)
            mode, base = "revise", state["candidates"][-1]["text"]
        return {"revision_notes": notes, "revise_mode": mode, "revise_base": base,
                "revisions_used": state.get("revisions_used", 0) + 1,
                "retry_count": state.get("retry_count", 0) + 1,
                "trace": [{"step": "revise", "mode": mode, "notes": notes}]}

    # ---- finalize -------------------------------------------------------
    def _rank(self, c: dict):
        score = c["critic_score"] if c.get("critic_score") is not None else self.cfg.pass_default_score
        return (len(c["errors"]), -score, len(c["warns"]), c["attempt"])

    def finalize(self, state: ScriptState):
        cands = state.get("candidates", [])
        if cands:
            best = min(cands, key=self._rank)
            final, ev = best["text"], {"step": "finalize", "chosen_attempt": best["attempt"], "candidates": len(cands),
                                       "errors": best["errors"], "critic_score": best.get("critic_score")}
        else:
            final, ev = state.get("draft", ""), {"step": "finalize", "legacy": True}
        if not final.strip():
            return self.fallback_scene(state)
        return {"final_script": final, "trace": [ev]}

    # ---- legacy nodes (unchanged behaviour) -----------------------------
    def rewrite_query(self, state: ScriptState):
        expanded = state["user_prompt"] + " details lore settings characters"
        return {"search_query": expanded, "retry_count": state.get("retry_count", 0) + 1,
                "trace": [{"step": "rewrite_query"}]}

    def fallback_scene(self, state: ScriptState):
        return {"final_script": "The AI could not find enough reliable continuity context in the Show Bible to draft this scene. "
                                "Please adjust your prompt to focus on existing characters and settings.",
                "sources": [], "trace": [{"step": "fallback"}]}

    # ------------------------------------------------------------------ GRAPH
    def _build_graph(self):
        g = StateGraph(ScriptState)
        g.add_node("retrieve", self.retrieve_context)
        g.add_node("draft_scene", self.draft_scene)
        g.add_node("validate", self.validate_draft)
        g.add_node("critic", self.critic_node)
        g.add_node("prepare_revision", self.prepare_revision)
        g.add_node("finalize", self.finalize)
        g.add_node("rewrite_query", self.rewrite_query)
        g.add_node("fallback", self.fallback_scene)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "draft_scene")
        g.add_conditional_edges("draft_scene", self.route_after_draft,
                                {"validate": "validate", "finalize": "finalize", "retry": "rewrite_query", "fallback": "fallback"})
        g.add_conditional_edges("validate", self.route_after_validate,
                                {"revise": "prepare_revision", "critic": "critic", "finalize": "finalize"})
        g.add_conditional_edges("critic", self.route_after_critic,
                                {"revise": "prepare_revision", "finalize": "finalize"})
        g.add_edge("prepare_revision", "draft_scene")
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("finalize", END)
        g.add_edge("fallback", END)
        return g.compile()

    # ------------------------------------------------------------------ PUBLIC API
    def ask_agent(self, user_prompt: str, k_chunks: int = 3, agentic: bool = True, use_critic: bool = True) -> dict:
        initial_state = {
            "user_prompt": user_prompt, "search_query": user_prompt, "retry_count": 0, "k_chunks": k_chunks,
            "agentic": agentic, "use_critic": use_critic, "attempt": 0, "revisions_used": 0,
            "candidates": [], "trace": [],
        }
        return self.graph.invoke(initial_state, config={"recursion_limit": 40})