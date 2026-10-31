import streamlit as st
from rag_engine import ScriptCraftRAG

st.set_page_config(page_title="ScriptCraft AI", layout="wide")
st.title("🎬 ScriptCraft: Continuity-Aware Screenwriter")

# Securely load the Groq API key from Streamlit Secrets
# Locally, store this in .streamlit/secrets.toml
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Cache the RAG initialization so FAISS doesn't rebuild on every button click
@st.cache_resource(show_spinner="Loading Show Bible and Models...")
def load_rag_engine():
    return ScriptCraftRAG(groq_api_key=GROQ_API_KEY)

rag_system = load_rag_engine()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Director's Chair")
    user_prompt = st.text_area("Describe the scene:", height=150, 
                               placeholder="Arthur is cleaning the espresso machine when Lena bursts in...")
    
    if st.button("Generate Scene", type="primary"):
        if user_prompt:
            with st.spinner("Drafting scene and checking continuity..."):
                try:
                    result = rag_system.generate_scene(user_prompt)
                    st.session_state['script'] = result['script']
                    st.session_state['context'] = result['retrieved_context']
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("Drafted Script")
    if 'script' in st.session_state:
        st.markdown(f"```text\n{st.session_state['script']}\n```")
        
        with st.expander("🔍 View Retrieved Continuity Context"):
            st.info(st.session_state['context'])
    else:
        st.info("Your generated script will appear here.")