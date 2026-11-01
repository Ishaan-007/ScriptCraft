import streamlit as st
from rag_engine import ScriptCraftRAG

st.set_page_config(page_title="ScriptCraft AI", layout="centered")
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
            except Exception as e:
                st.error(f"Error: {e}")

# --- BOTTOM SECTION: OUTPUT ---
if 'script' in st.session_state:
    st.divider()
    
    # Show metadata and retry status
    retry_msg = f"(Draft rejected & retried {st.session_state['retry_count']} times)" if st.session_state['retry_count'] > 0 else "(First draft approved)"
    
    if st.session_state['sources']:
        sources_str = ", ".join(st.session_state['sources'])
        st.caption(f"🧠 **Lore Retrieved From:** `{sources_str}` | {retry_msg}")
    
    st.subheader("📜 Drafted Script")
    
    script_html = f"""
    <div style="
        font-family: 'Courier New', Courier, monospace; 
        white-space: pre-wrap; 
        background-color: #1e1e1e; 
        padding: 20px; 
        border-radius: 8px; 
        color: #e0e0e0;
        line-height: 1.4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    ">
{st.session_state['script']}
    </div>
    """
    st.markdown(script_html, unsafe_allow_html=True)
    
    if st.session_state['context']:
        with st.expander("🔍 View Exact Context Chunks Used"):
            st.info("The Agent pulled these exact chunks to maintain continuity:")
            st.write(st.session_state['context'])