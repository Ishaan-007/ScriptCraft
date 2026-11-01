import streamlit as st
from rag_engine import ScriptCraftRAG

# Use a wide layout to maximize room for screenplays
st.set_page_config(page_title="ScriptCraft AI", layout="wide")
st.title("🎬 ScriptCraft: Intelligent Screenplay-writer Assistant")

# Securely load the Groq API key from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml).")
    st.stop()

@st.cache_resource(show_spinner="Loading Show Bible, FAISS, and LangGraph...")
def load_rag_engine():
    return ScriptCraftRAG(groq_api_key=GROQ_API_KEY)

rag_system = load_rag_engine()

# Custom CSS injected to display the generated text completely unparsed and raw,
# while safely honoring screen-wide boundaries and word-wrapping.
st.markdown("""
    <style>
    .screenplay-container {
        background-color: #11141a;
        padding: 30px;
        border-radius: 8px;
        border: 1px solid #2d3139;
        width: 100% !important;
    }
    .raw-script-pre {
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: #e0e0e0 !important;
        white-space: pre-wrap !important;       /* Preserves text line breaks & spaces */
        word-wrap: break-word !important;       /* Breaks long continuous blocks */
        overflow-x: hidden !important;          /* Disables annoying horizontal scrolling */
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main 2-column layout to naturally provide more space for script content
main_col, side_col = st.columns([3, 1])

with main_col:
    # --- TOP SECTION: INPUT ---
    st.subheader("🎬 Director's Chair")
    user_prompt = st.text_area(
        "Describe the scene:", 
        height=120, 
        placeholder="e.g., Silas Vane walks into The Rusted Mug to threaten Arthur, but Jax is hiding under a table..."
    )

    k_slider = st.slider(
        "Number of continuity chunks to retrieve (k):", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="Higher values pull more lore from the Show Bible but may dilute the specific focus of the scene."
    )

    if st.button("Generate Scene", type="primary", use_container_width=True):
        if user_prompt:
            with st.spinner("Agent is retrieving lore, drafting, and reviewing continuity..."):
                try:
                    result = rag_system.ask_agent(user_prompt, k_chunks=k_slider)
                    st.session_state['script'] = result.get('final_script', '')
                    st.session_state['context'] = result.get('context', '')
                    st.session_state['sources'] = result.get('sources', [])
                    st.session_state['retry_count'] = result.get('retry_count', 0)
                    st.session_state['reasoning'] = result.get('reasoning', '')
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- BOTTOM SECTION: OUTPUT ---
    if 'script' in st.session_state:
        st.divider()
        
        # 1. Display Chain of Thought if available
        if st.session_state.get('reasoning'):
            with st.expander("🧠 View Agent's Chain of Thought / Scriptwriting Logic", expanded=True):
                st.info(st.session_state['reasoning'])
        else:
            st.warning("Note: The model did not return separate reasoning tokens. If output remains instant, verify parameters inside your Groq console dashboard.")
                
        # 2. Main Screenplay Raw Unparsed Output Box
        st.subheader("📝 Drafted Screenplay")
        with st.container():
            st.markdown('<div class="screenplay-container">', unsafe_allow_html=True)
            # st.text element escapes markdown tokens entirely, displaying raw symbols like **, *, or --- exactly as generated
            st.text(st.session_state['script'])
            st.markdown('</div>', unsafe_allow_html=True)

# Diagnostics moved directly into the parallel side column layout rather than the narrow sidebar 
with side_col:
    st.subheader("🔍 Diagnostics")
    if 'script' in st.session_state:
        st.write(f"**Retries Made:** {st.session_state.get('retry_count', 0)}")
        
        if st.session_state.get('sources'):
            st.write("**Referenced Show Bible Files:**")
            for source in st.session_state['sources']:
                st.caption(f"📄 {source}")
        else:
            st.caption("No sources matched continuity.")
            
        with st.expander("Show Bible Vector Context", expanded=False):
            st.markdown(st.session_state.get('context', 'No context loaded.'))
