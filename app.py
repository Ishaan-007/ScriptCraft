import streamlit as st
from rag_engine import ScriptCraftRAG

st.set_page_config(page_title="ScriptCraft AI", layout="centered")
st.title("🎬 ScriptCraft: Continuity-Aware Screenwriter")

# Securely load the Groq API key from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml).")
    st.stop()

@st.cache_resource(show_spinner="Loading Show Bible and FAISS Index...")
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

if st.button("Generate Scene", type="primary", use_container_width=True):
    if user_prompt:
        with st.spinner("Retrieving continuity context and drafting scene..."):
            try:
                result = rag_system.generate_scene(user_prompt)
                st.session_state['script'] = result['script']
                st.session_state['context'] = result['retrieved_context']
            except Exception as e:
                st.error(f"Error: {e}")

# --- BOTTOM SECTION: OUTPUT ---
if 'script' in st.session_state:
    st.divider()
    st.subheader("📜 Drafted Script")
    
    # CSS injection for proper Screenplay formatting with word-wrap
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
    ">
{st.session_state['script']}
    </div>
    """
    st.markdown(script_html, unsafe_allow_html=True)
    
    st.write("") # Spacer
    with st.expander("🔍 View Retrieved Continuity Context"):
        st.info("The AI pulled the following specific lore from your Show Bible to write this scene:")
        st.write(st.session_state['context'])