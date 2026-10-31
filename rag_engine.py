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
        self.bible_path = self.base_dir / "show_bible.txt"
        
        # Initialize Groq Client
        self.client = Groq(api_key=groq_api_key)
        
        # Load and Chunk the Show Bible
        bible_text = self.bible_path.read_text(encoding="utf-8")
        doc = Document(page_content=bible_text, metadata={"source": "show_bible.txt"})
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc])
        
        # Create FAISS Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

    def generate_scene(self, user_prompt: str) -> dict:
        # Retrieve continuity context
        docs = self.retriever.invoke(user_prompt)
        context = "\n\n".join(d.page_content for d in docs)
        
        # Strict Screenplay Prompt
        system_prompt = f"""You are an expert Hollywood screenwriter. 
Write a scene based on the USER PROMPT. 
You MUST use the CONTINUITY CONTEXT to ensure character personalities and settings are accurate.
Format the output STRICTLY in standard screenplay format. Do not include conversational filler.

CONTINUITY CONTEXT:
{context}"""

        # Call Groq API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.7,
        )
        
        return {
            "script": chat_completion.choices[0].message.content,
            "retrieved_context": context
        }