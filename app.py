import html
import streamlit as st
from rag_engine import ScriptCraftRAG

st.set_page_config(page_title="ScriptCraft AI", layout="centered")
st.title("🎬 ScriptCraft: Intelligent Screenplay-writer Assistant")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml).")
    st.stop()


@st.cache_resource(show_spinner="Loading Show Bible, FAISS, and LangGraph...")
def load_rag_engine():
    return ScriptCraftRAG(groq_api_key=GROQ_API_KEY)


rag_system = load_rag_engine()

# ======================================================================
# Workflow visualisation (Graphviz DOT, rendered natively by Streamlit)
# ======================================================================
STEP_TO_NODE = {"retrieve": "retrieve", "draft": "draft_scene", "validate": "validate", "critic": "critic",
                "revise": "prepare_revision", "rewrite_query": "rewrite_query", "finalize": "finalize",
                "fallback": "fallback"}
GREEN, ORANGE, RED, BLUE, GREY = "#c8e6c9", "#ffe0b2", "#ffcdd2", "#dbeafe", "#f1f1f1"
HEAD = ('bgcolor="transparent"; rankdir=TB; nodesep=0.35; ranksep=0.45; '
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.15,0.08", color="#9e9e9e"]; '
        'edge [fontname="Helvetica", fontsize=9, color="#9e9e9e", arrowsize=0.7];')


def _esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _label(*lines):
    """Join lines into a DOT label (real newlines in -> DOT \n out)."""
    flat = [x for line in lines for x in str(line).split("\n") if x]
    return "\\n".join(_esc(x) for x in flat)


def _chunks(ev):
    n = ev.get("chunks", 0)
    return f"{n} chunk{'s' if n != 1 else ''}" + (f" + {ev['boosted']} boosted" if ev.get("boosted") else "")


def _short(model):
    return model.split("/")[-1]


def _node_status(node, evs):
    """-> (fill colour, '×N' suffix, subtitle) for a node that was visited."""
    times = f" ×{len(evs)}" if len(evs) > 1 else ""
    if node == "validate":
        bad = [e for e in evs if e.get("errors")]
        sub = (f"✗ flagged {len(bad)}/{len(evs)}: " + ", ".join(bad[-1]["errors"][:2])) if bad else "✓ passed"
        return (ORANGE if bad else GREEN), times, sub
    if node == "critic":
        scored = [e for e in evs if not e.get("skipped")]
        if not scored:
            return GREY, times, "skipped"
        bad = [e for e in scored if e.get("major")]
        sub = "score " + " → ".join(str(e["score"]) for e in scored) + " /10"
        return (ORANGE if bad else GREEN), times, sub
    if node in ("prepare_revision", "rewrite_query"):
        return ORANGE, times, ", ".join(sorted({e.get("mode", "retry") for e in evs}))
    if node == "fallback":
        return RED, times, ""
    if node == "finalize":
        e = evs[-1]
        return GREEN, times, (f"draft #{e['chosen_attempt']}" if e.get("chosen_attempt") else "legacy")
    if node == "draft_scene":
        return (RED if any(e.get("error") for e in evs) else BLUE), times, f"{evs[-1].get('words', '?')} words"
    e = evs[-1]                                              # retrieve
    return BLUE, times, _chunks(e)


def workflow_dot(graph, trace, cfg):
    """The REAL compiled LangGraph topology, with the path this run took highlighted."""
    g = graph.get_graph()
    titles = {"retrieve": ("Retrieve", "FAISS + entity boost"), "draft_scene": ("Draft", _short(cfg.drafter_model)),
              "validate": ("Rule validator", "+ auto-repair"), "critic": ("Continuity critic", _short(cfg.critic_model)),
              "prepare_revision": ("Prepare revision", ""), "finalize": ("Finalize", "best draft wins"),
              "rewrite_query": ("Rewrite query", ""), "fallback": ("Fallback", "")}
    visits = {}
    for e in trace:
        if e["step"] in STEP_TO_NODE:
            visits.setdefault(STEP_TO_NODE[e["step"]], []).append(e)
    path = ["__start__"] + [STEP_TO_NODE[e["step"]] for e in trace if e["step"] in STEP_TO_NODE] + ["__end__"]
    used = {}
    for i, pair in enumerate(zip(path, path[1:]), 1):
        used.setdefault(pair, []).append(i)

    out = ["digraph W {", HEAD]
    for nid in g.nodes:
        if nid in ("__start__", "__end__"):
            text = "START" if nid == "__start__" else "END"
            out.append(f'{nid} [label="{text}", shape=circle, fillcolor="#424242", fontcolor="white", '
                       f'color="#424242", width=0.6, fixedsize=true];')
            continue
        title, static = titles.get(nid, (nid, ""))
        if nid in visits:
            fill, times, sub = _node_status(nid, visits[nid])
            label = _label(title + times, static, sub)
            attrs = f'fillcolor="{fill}", fontcolor="#1b1b1b", color="#2e7d32", penwidth=2'
        else:
            label = _label(title, static)
            attrs = f'fillcolor="{GREY}", fontcolor="#9a9a9a", color="#d0d0d0"'
        out.append(f'{nid} [label="{label}", {attrs}];')
    for ed in g.edges:
        route = _esc(ed.data) if ed.data else ""
        if (ed.source, ed.target) in used:
            order = ", ".join(f"#{n}" for n in used[(ed.source, ed.target)])
            text = (route + " " if route else "") + order
            out.append(f'{ed.source} -> {ed.target} [color="#2e7d32", penwidth=2.6, fontcolor="#2e7d32", label="{text}"];')
        else:
            out.append(f'{ed.source} -> {ed.target} [style=dashed, color="#cfcfcf", fontcolor="#b0b0b0", label="{route}"];')
    out.append("}")
    return "\n".join(out)


def _describe(ev):
    """One trace event -> (title, subtitle, fill colour) for the timeline."""
    s = ev["step"]
    if s == "retrieve":
        return "Retrieve", _chunks(ev), BLUE
    if s == "draft":
        if ev.get("error"):
            return "Draft", f"failed ({ev['error']})", RED
        return f"Draft #{ev['attempt']}", f"{ev['words']} words · {ev['mode']}" + (" · ⚠ cut off" if ev.get("finish") == "length" else ""), BLUE
    if s == "validate":
        fix = f"auto-fixed {len(ev['repairs'])}×" if ev["repairs"] else ""
        if ev["errors"]:
            return "Rule validator", "✗ " + ", ".join(ev["errors"][:3]) + ("\n" + fix if fix else ""), ORANGE
        return "Rule validator", "✓ passed" + (f" ({fix})" if fix else ""), GREEN
    if s == "critic":
        if ev.get("skipped"):
            return "Critic", "skipped (unavailable)", GREY
        if ev["major"]:
            kinds = ", ".join(sorted({m.get("type") or "?" for m in ev["major"]}))
            return f"Critic {ev['score']}/10", f"{len(ev['major'])} major: {kinds}", ORANGE
        return f"Critic {ev['score']}/10", "no major issues", GREEN
    if s == "revise":
        return "Revision", ev["mode"], ORANGE
    if s == "rewrite_query":
        return "Rewrite query", "context was insufficient", ORANGE
    if s == "finalize":
        if ev.get("chosen_attempt"):
            return "Final script", f"draft #{ev['chosen_attempt']}" + (f" · critic {ev['critic_score']}/10" if ev.get("critic_score") else ""), GREEN
        return "Final script", "legacy single-shot", GREEN
    return "Fallback", "no reliable context", RED


def timeline_dot(trace):
    """The run unrolled step by step (every retry becomes its own box)."""
    out = ["digraph T {", HEAD]
    for i, ev in enumerate(trace):
        title, sub, fill = _describe(ev)
        extra = ', penwidth=2.2, color="#2e7d32"' if ev["step"] == "finalize" else ""
        out.append(f'n{i} [label="{_label(title, sub)}", fillcolor="{fill}"{extra}];')
        if i:
            out.append(f"n{i - 1} -> n{i};")
    out.append("}")
    return "\n".join(out)


def show_dot(dot):
    try:
        st.graphviz_chart(dot, width="stretch")                  # current Streamlit
    except TypeError:
        st.graphviz_chart(dot, use_container_width=True)         # older Streamlit


# ======================================================================
# SIDEBAR: agent controls
# ======================================================================
with st.sidebar:
    st.header("⚙️ Agent settings")
    agentic = st.toggle("Agentic quality checks", value=True,
                        help="Off = original single-shot pipeline (no validation, no critic, no retries).")
    use_critic = st.toggle("LLM continuity critic", value=True, disabled=not agentic,
                           help=f"Second model ({rag_system.cfg.critic_model}) checks the draft against the Show Bible.")
    st.caption(f"Drafter: `{rag_system.cfg.drafter_model}`  \nCritic: `{rag_system.cfg.critic_model}`")

# ======================================================================
# INPUT
# ======================================================================
st.subheader("🎬 Director's Chair")
user_prompt = st.text_area(
    "Describe the scene:", height=120,
    placeholder="e.g., Silas Vane walks into The Rusted Mug to threaten Arthur, but Jax is hiding under a table...")

k_slider = st.slider(
    "Number of continuity chunks to retrieve (k):", min_value=1, max_value=5, value=3,
    help="Higher values pull more lore from the Show Bible but may dilute the specific focus of the scene.")

if st.button("Generate Scene", type="primary", use_container_width=True):
    if user_prompt:
        with st.spinner("Agent is retrieving lore, drafting, validating and reviewing continuity..."):
            try:
                result = rag_system.ask_agent(user_prompt, k_chunks=k_slider, agentic=agentic, use_critic=use_critic)
                st.session_state["script"] = result.get("final_script", "")
                st.session_state["context"] = result.get("context", "")
                st.session_state["sources"] = result.get("sources", [])
                st.session_state["retry_count"] = result.get("retry_count", 0)
                st.session_state["trace"] = result.get("trace", [])
                st.session_state["candidates"] = result.get("candidates", [])
            except Exception as e:
                st.error(f"Error: {e}")

# ======================================================================
# OUTPUT
# ======================================================================
if "script" in st.session_state:
    st.divider()

    n = st.session_state["retry_count"]
    retry_msg = f"(Draft rejected & revised {n} time{'s' if n != 1 else ''})" if n > 0 else "(First draft approved)"
    if st.session_state["sources"]:
        st.caption(f"🧠 **Lore Retrieved From:** `{', '.join(st.session_state['sources'])}` | {retry_msg}")

    st.subheader("📜 Drafted Script")
    
    # Check if we have multiple candidates to show tabs
    cands = st.session_state.get("candidates", [])
    if len(cands) > 1:
        tabs = st.tabs(["🌟 Final Best Draft"] + [f"Draft {c['attempt']}" for c in cands])
        
        with tabs[0]:
            script_html = (f"<div style=\"font-family:'Courier New',Courier,monospace;white-space:pre-wrap;background-color:#1e1e1e;"
                           f"padding:20px;border-radius:8px;color:#e0e0e0;line-height:1.4;box-shadow:0 4px 6px rgba(0,0,0,0.3);"
                           f"margin-bottom:20px;\">{html.escape(st.session_state['script'])}</div>")
            st.markdown(script_html, unsafe_allow_html=True)
            
        for i, c in enumerate(cands):
            with tabs[i + 1]:
                # --- 1. CONCISE TOP BAR ---
                score = c.get("critic_score")
                score_text = f"🧠 **Critic:** {score}/10" if score else "🧠 **Critic:** N/A"
                
                errs = c.get("errors", [])
                fmt_text = f"❌ **Format:** Fail ({', '.join(errs)})" if errs else "✅ **Format:** Pass"
                
                repairs = c.get("repairs", [])
                rep_text = f" | 🔧 **Repairs:** {len(repairs)}" if repairs else ""
                
                st.markdown(f"{score_text} &nbsp;|&nbsp; {fmt_text}{rep_text}")
                
                # --- 2. SCRIPT DISPLAY ---
                cand_html = (f"<div style=\"font-family:'Courier New',Courier,monospace;white-space:pre-wrap;background-color:#2a2a2a;"
                             f"padding:20px;border-radius:8px;color:#bdbdbd;line-height:1.4;"
                             f"margin-bottom:10px;\">{html.escape(c.get('text', ''))}</div>")
                st.markdown(cand_html, unsafe_allow_html=True)
                
                # --- 3. DETAILED CRITIC NOTES (BOTTOM) ---
                viols = c.get("violations", [])
                if viols:
                    with st.expander("📝 Critic's Continuity Notes", expanded=False):
                        for v in viols:
                            st.markdown(f"**{str(v.get('type', 'Issue')).title()}**")
                            st.markdown(f"- **Show Bible Context:** *\"{v.get('evidence', 'N/A')}\"*")
                            st.markdown(f"- **Draft Error:** {v.get('problem', 'N/A')}")
                            st.markdown(f"- **Suggested Fix:** {v.get('fix', 'N/A')}")
                            st.divider()
    else:
        # Standard single-view fallback
        script_html = (f"<div style=\"font-family:'Courier New',Courier,monospace;white-space:pre-wrap;background-color:#1e1e1e;"
                       f"padding:20px;border-radius:8px;color:#e0e0e0;line-height:1.4;box-shadow:0 4px 6px rgba(0,0,0,0.3);"
                       f"margin-bottom:20px;\">{html.escape(st.session_state['script'])}</div>")
        st.markdown(script_html, unsafe_allow_html=True)
    st.download_button("⬇️ Download as .fountain", st.session_state["script"],
                       file_name="scene.fountain", mime="text/plain")

    # ---- dynamic workflow graph ----
    if st.session_state.get("trace"):
        st.subheader("🧭 How the agent got here")
        tab_map, tab_run = st.tabs(["Workflow map (path highlighted)", "Run timeline"])
        with tab_map:
            show_dot(workflow_dot(rag_system.graph, st.session_state["trace"], rag_system.cfg))
            st.caption("🟩 passed · 🟧 problem found / revision · 🟦 step ran · ⬜ not visited this run · "
                       "green arrows = path taken (#n = order). Drawn from the compiled LangGraph itself.")
        with tab_run:
            show_dot(timeline_dot(st.session_state["trace"]))

# ---- SIDEBAR (filled after generation so it always shows the latest run) ----
if st.session_state.get("context"):
    with st.sidebar:
        with st.expander("🔍 Retrieved context chunks", expanded=False):
            st.caption("The agent pulled these exact chunks to maintain continuity:")
            st.markdown(st.session_state["context"])