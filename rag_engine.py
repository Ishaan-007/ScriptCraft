import os
from pathlib import Path
from typing import TypedDict, List
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

# Define the shared state for the LangGraph workflow
class ScriptState(TypedDict, total=False):
    user_prompt: str
    search_query: str
    context: str
    sources: List[str]
    draft: str
    retry_count: int
    final_script: str
    k_chunks: int

class ScriptCraftRAG:
    def __init__(self, groq_api_key: str):
        self.base_dir = Path(__file__).resolve().parent
        self.bible_dir = self.base_dir / "show_bible"
        
        # Initialize Groq Client
        self.client = Groq(api_key=groq_api_key)
        self.model_name = "openai/gpt-oss-20b"
        
        # Load and Chunk Show Bible
        docs = []
        for file_path in self.bible_dir.glob("*.md"):
            text = file_path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": file_path.name}))
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Create FAISS Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Compile the LangGraph Workflow
        self.graph = self._build_graph()

    # --- LANGGRAPH NODES ---

    def retrieve_context(self, state: ScriptState):
        k = state.get("k_chunks", 3)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        search_query = state.get("search_query", state["user_prompt"])
        retrieved_docs = retriever.invoke(search_query)
        
        context_parts = []
        unique_sources = set()
        
        for d in retrieved_docs:
            source_file = d.metadata.get("source", "unknown file")
            unique_sources.add(source_file)
            context_parts.append(f"### Source: {source_file}\n{d.page_content.strip()}")
        
        return {
            "context": "\n\n".join(context_parts),
            "sources": list(unique_sources)
        }

    def draft_scene(self, state: ScriptState):
        system_prompt = f"""You are an expert Hollywood screenwriter. 
Write a scene based on the USER PROMPT. 
You MUST use the CONTINUITY CONTEXT to ensure character personalities, lore, and settings are accurate.

RULES:
- If the context does not contain enough information to write the requested scene accurately, output exactly: INSUFFICIENT_CONTEXT
- Otherwise, output STRICTLY in standard Fountain/Screenplay format.
- Scene Headings must be ALL CAPS (e.g., INT. COFFEE SHOP - DAY).
- Character names above dialogue must be ALL CAPS and centered (or on their own line).
- Do NOT include conversational filler, markdown formatting blocks like ``` or citations in the text. Just the raw script.

CONTINUITY CONTEXT:
{state.get("context", "")}"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["user_prompt"]}
            ],
            model=self.model_name, 
            temperature=0.7,
        )
        
        raw_script = chat_completion.choices[0].message.content.strip()
        if raw_script.startswith("```"):
            raw_script = "\n".join(raw_script.split("\n")[1:])
        if raw_script.endswith("```"):
            raw_script = raw_script[:-3].strip()
            
        return {"draft": raw_script}

    def review_draft(self, state: ScriptState):
        draft = state.get("draft", "").strip()
        if "INSUFFICIENT_CONTEXT" in draft.upper():
            return {"final_script": ""}
        return {"final_script": draft}

    def route_after_review(self, state: ScriptState):
        if state.get("final_script"):
            return "finish"
        if state.get("retry_count", 0) < 1:
            return "retry"
        return "fallback"

    def rewrite_query(self, state: ScriptState):
        # Expand the query slightly for a broader retrieval on retry
        expanded = state["user_prompt"] + " details lore settings characters"
        return {
            "search_query": expanded,
            "retry_count": state.get("retry_count", 0) + 1
        }

    def fallback_scene(self, state: ScriptState):
        return {
            "final_script": "The AI could not find enough reliable continuity context in the Show Bible to draft this scene. Please adjust your prompt to focus on existing characters and settings.",
            "sources": []
        }

    # --- BUILD GRAPH ---

    def _build_graph(self):
        graph_builder = StateGraph(ScriptState)
        
        graph_builder.add_node("retrieve", self.retrieve_context)
        graph_builder.add_node("draft_scene", self.draft_scene)
        graph_builder.add_node("review_draft", self.review_draft)
        graph_builder.add_node("rewrite_query", self.rewrite_query)
        graph_builder.add_node("fallback", self.fallback_scene)
        
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "draft_scene")
        graph_builder.add_edge("draft_scene", "review_draft")
        
        graph_builder.add_conditional_edges(
            "review_draft",
            self.route_after_review,
            {
                "finish": END,
                "retry": "rewrite_query",
                "fallback": "fallback"
            }
        )
        
        graph_builder.add_edge("rewrite_query", "retrieve")
        graph_builder.add_edge("fallback", END)
        
        return graph_builder.compile()

    # --- PUBLIC API ---

    def ask_agent(self, user_prompt: str, k_chunks: int = 3) -> dict:
        initial_state = {
            "user_prompt": user_prompt,
            "search_query": user_prompt,
            "retry_count": 0,
            "k_chunks": k_chunks
        }
        return self.graph.invoke(initial_state)