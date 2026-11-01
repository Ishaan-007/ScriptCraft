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

# --- SIDEBAR: agent controls (handy for live A/B demos) ---
with st.sidebar:
    st.header("⚙️ Agent settings")
    agentic = st.toggle("Agentic quality checks", value=True,
                        help="Off = original single-shot pipeline (no validation, no critic, no retries).")
    use_critic = st.toggle("LLM continuity critic", value=True, disabled=not agentic,
                           help=f"Second model ({rag_system.cfg.critic_model}) checks the draft against the Show Bible.")
    st.caption(f"Drafter: `{rag_system.cfg.drafter_model}`  \nCritic: `{rag_system.cfg.critic_model}`")

# --- TOP SECTION: INPUT ---
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
            except Exception as e:
                st.error(f"Error: {e}")

# --- BOTTOM SECTION: OUTPUT ---
if "script" in st.session_state:
    st.divider()

    n = st.session_state["retry_count"]
    retry_msg = f"(Draft rejected & revised {n} time{'s' if n != 1 else ''})" if n > 0 else "(First draft approved)"
    if st.session_state["sources"]:
        st.caption(f"🧠 **Lore Retrieved From:** `{', '.join(st.session_state['sources'])}` | {retry_msg}")

    st.subheader("📜 Drafted Script")
    # st.html keeps blank lines intact (markdown HTML blocks end at the first blank line); escape the script text.
    script_html = (
        "<div style=\"font-family:'Courier New',Courier,monospace;white-space:pre-wrap;background-color:#1e1e1e;"
        "padding:20px;border-radius:8px;color:#e0e0e0;line-height:1.4;box-shadow:0 4px 6px rgba(0,0,0,0.3);"
        f"margin-bottom:20px;\">{html.escape(st.session_state['script'])}</div>")
    if hasattr(st, "html"):
        st.html(script_html)
    else:
        st.markdown(script_html, unsafe_allow_html=True)
    st.download_button("⬇️ Download as .fountain", st.session_state["script"],
                       file_name="scene.fountain", mime="text/plain")

    if st.session_state.get("trace"):
        with st.expander("🕵️ Agent trace (what the workflow did)"):
            for ev in st.session_state["trace"]:
                step = ev["step"]
                if step == "retrieve":
                    st.markdown(f"**Retrieve** — {ev['chunks']} chunks" + (f" + {ev['boosted']} boosted for named characters/places" if ev.get("boosted") else ""))
                elif step == "draft":
                    st.markdown(f"**Draft #{ev['attempt']}** ({ev.get('mode', 'initial')}) — {ev.get('words', 0)} words, finish=`{ev.get('finish')}`" if "words" in ev else f"**Draft** — failed (`{ev.get('error')}`)")
                elif step == "validate":
                    status = "✅ passed" if not ev["errors"] else f"❌ {', '.join(ev['errors'])}"
                    rep = f" · auto-repaired: {'; '.join(ev['repairs'])}" if ev["repairs"] else ""
                    warn = f" · ⚠️ {', '.join(ev['warns'])}" if ev["warns"] else ""
                    st.markdown(f"**Rule validator** — {status}{rep}{warn}")
                elif step == "critic":
                    if ev.get("skipped"):
                        st.markdown("**Continuity critic** — skipped (model unavailable)")
                    else:
                        st.markdown(f"**Continuity critic** (`{ev['model']}`) — score {ev['score']}/10, {len(ev['major'])} major issue(s)")
                        for v in ev["major"]:
                            st.markdown(f"  - *{v.get('type')}*: context says “{v.get('evidence')}” → {v.get('problem')}")
                elif step == "revise":
                    st.markdown(f"**Revision requested** ({ev['mode']})")
                elif step == "finalize":
                    st.markdown(f"**Final** — chose draft #{ev.get('chosen_attempt', '—')}" + (f", critic score {ev['critic_score']}" if ev.get("critic_score") else ""))
                else:
                    st.markdown(f"**{step}**")

    if st.session_state["context"]:
        with st.expander("🔍 View Exact Context Chunks Used"):
            st.info("The Agent pulled these exact chunks to maintain continuity:")
            st.write(st.session_state["context"])