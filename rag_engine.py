import os
from pathlib import Path
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ScriptCraftRAG:
    def __init__(self, groq_api_key: str):
        self.base_dir = Path(__file__).resolve().parent
        self.bible_dir = self.base_dir / "show_bible"
        
        # Initialize Groq Client
        self.client = Groq(api_key=groq_api_key)
        
        # Load all markdown files from the show_bible directory
        docs = []
        for file_path in self.bible_dir.glob("*.md"):
            text = file_path.read_text(encoding="utf-8")
            # Attach the filename as metadata for citations
            docs.append(Document(page_content=text, metadata={"source": file_path.name}))
        
        # Chunking strategy: 600 characters with a 100-character overlap
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create FAISS Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Retrieve the 3 most relevant chunks
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def generate_scene(self, user_prompt: str) -> dict:
        # 1. Retrieve continuity context
        retrieved_docs = self.retriever.invoke(user_prompt)
        
        # 2. Format the context and inject metadata (source files)
        context_parts = []
        for d in retrieved_docs:
            source_file = d.metadata.get("source", "unknown file")
            context_parts.append(f"\n{d.page_content}")
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        # 3. Strict Screenplay Prompt
        system_prompt = f"""You are an expert Hollywood screenwriter. 
Write a scene based on the USER PROMPT. 
You MUST use the CONTINUITY CONTEXT to ensure character personalities, lore, and settings are accurate.

FORMATTING RULES:
- Output STRICTLY in standard Fountain/Screenplay format.
- Scene Headings must be ALL CAPS (e.g., INT. COFFEE SHOP - DAY).
- Character names above dialogue must be ALL CAPS and centered (or on their own line).
- Parentheticals should be lowercase in brackets under the name (e.g., (sighs)).
- Do NOT include conversational filler, markdown formatting blocks like ```text, or introductory text. Just the script.

CONTINUITY CONTEXT:
{context_string}"""

        # 4. Call Groq API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.7,
        )
        
        return {
            "script": chat_completion.choices[0].message.content.strip(),
            "retrieved_context": context_string
        }