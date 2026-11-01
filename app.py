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
                st.session_state['reasoning'] = result.get('reasoning', '')  # Saved to session state
            except Exception as e:
                st.error(f"Error: {e}")

# --- BOTTOM SECTION: OUTPUT ---
if 'script' in st.session_state:
    st.divider()
    
    # 1. Display Chain of Thought if available
    if st.session_state.get('reasoning'):
        with st.expander("🧠 View Agent's Chain of Thought / Scriptwriting Logic", expanded=False):
            st.info(st.session_state['reasoning'])
            
    # 2. Main Screenplay Code Block / Output
    st.subheader("📝 Drafted Screenplay")
    st.code(st.session_state['script'], language="markdown")
    
    # 3. Sidebar Metadata (Sources / RAG Context Used)
    with st.sidebar:
        st.subheader("🔍 Retrieval Diagnostics")
        st.write(f"**Retries Made:** {st.session_state.get('retry_count', 0)}")
        
        if st.session_state.get('sources'):
            st.write("**Referenced Show Bible Files:**")
            for source in st.session_state['sources']:
                st.caption(f"📄 {source}")
        else:
            st.caption("No sources matched continuity.")
            
        with st.expander("Raw Injected Vector Context"):
            st.text(st.session_state.get('context', 'No context loaded.'))
