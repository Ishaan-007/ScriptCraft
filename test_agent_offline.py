"""
Offline wiring test: mocks Groq + the vector store, so it needs NO API key, NO GPU, NO embeddings download.
Run:  python test_agent_offline.py
"""
import types, httpx, groq
from langchain_core.documents import Document
from rag_engine import ScriptCraftRAG, AgentConfig
from script_tools import repair_script, validate_script, build_registry

GOOD = """INT. THE RUSTED MUG COFFEE SHOP - NIGHT

Neon flickers over cracked tiles. The espresso machine hisses and dies. ARTHUR PENDELTON wipes a mug, his left sleeve oddly stiff.

LENA VOSS pushes through the door, breathless.

LENA
(fast, bright)
Espresso. And the rumor that you used to chase corporate ghosts.

ARTHUR
Machine's broken.

LENA
Everything here is. Even your left hand hasn't moved in ten minutes.

Arthur sets the mug down. Metal clicks against ceramic.

ARTHUR
Short answers. Fewer questions.

LENA
Aegis Syndicate. Silas Vane. Ring a bell?

ARTHUR
That name cost me an arm.

He pushes up the sleeve. Chrome catches the neon.

LENA
(whispering)
Military grade.

ARTHUR
Now you know why I trust no one.

LENA
Then trust me for one cup of coffee.

Arthur finally reaches for the broken machine and slaps its side. It coughs steam. Lena laughs. For the first time in years, so does he.""" + (" The neon hums." * 40)

BAD_BLANKS = GOOD.replace("LENA\n(fast, bright)\nEspresso.", "LENA\n\n(fast, bright)\n\nEspresso.").replace("ARTHUR\nMachine's", "Arthur\n\nMachine's")
LOOP = "INT. THE RUSTED MUG COFFEE SHOP - NIGHT\n\nArthur wipes a mug.\n\n" + "\n\n".join(
    ["LENA\n(leans in)\nYou are hiding it. You are hiding the arm.", "ARTHUR\n(stares at her)\nI am not going to talk to you."] * 6)
RIGHT_ARM = GOOD.replace("left sleeve", "right sleeve").replace("your left hand", "your right hand")

class FakeClient:
    """Scripted responses per model name; records every call."""
    def __init__(self, script): self.script, self.calls = script, []
        
    @property
    def chat(self): return types.SimpleNamespace(completions=self)
    def create(self, **kw):
        self.calls.append(kw["model"])
        q = self.script[kw["model"]]
        item = q.pop(0)
        if isinstance(item, Exception): raise item
        text, finish = item if isinstance(item, tuple) else (item, "stop")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)])

class FakeStore:
    def __init__(self, docs): self.docs = docs
    def as_retriever(self, search_kwargs): 
        k = search_kwargs["k"]; return types.SimpleNamespace(invoke=lambda q: self.docs[:k])

def make(script, **cfg):
    docs = [Document(page_content="ARTHUR PENDELTON\nArthur is a 45-year-old former detective ... hides a military-grade cybernetic left arm.", metadata={"source": "characters.md"}),
            Document(page_content="THE RUSTED MUG COFFEE SHOP\nA dingy, neon-lit coffee shop ... Arthur's safe haven.", metadata={"source": "settings.md"})]
    return ScriptCraftRAG("x", AgentConfig(**cfg), client=FakeClient(script), vector_store=FakeStore(docs), bible_dir="sample_bible_for_tests")

D, C = "openai/gpt-oss-20b", "openai/gpt-oss-120b"
PASS = '{"score": 9, "violations": []}'
ARM = ('{"score": 4, "violations": [{"type":"continuity","severity":"major","evidence":"hides a military-grade cybernetic left arm",'
       '"problem":"draft says right arm","fix":"use the left arm"}]}')
PROMPT = "Lena walks into Arthur's coffee shop and hints about his arm. Silas Vane is mentioned."

def run(name, script, expect_calls, expect_chosen=None, **kw):
    rag = make(script, **cfg_for(kw)); r = rag.ask_agent(PROMPT, k_chunks=1, **{k:v for k,v in kw.items() if k in("agentic","use_critic")})
    calls = rag.client.calls
    steps = [e["step"] for e in r["trace"]]
    fin = [e for e in r["trace"] if e["step"] == "finalize"][0]
    ok = (calls == expect_calls) and (expect_chosen is None or fin.get("chosen_attempt") == expect_chosen)
    print(("PASS" if ok else "FAIL"), name, "| calls:", [c.split('/')[-1] for c in calls], "| steps:", steps, "| retry_count:", r["retry_count"])
    if not ok: print("   expected", expect_calls, expect_chosen, "got chosen", fin)
    return r

def cfg_for(kw): return {k: v for k, v in kw.items() if k not in ("agentic", "use_critic")}

# 1 happy path: 1 draft + 1 critic
r = run("happy path", {D: [GOOD], C: [PASS]}, [D, C], 1)
assert "Silas Vane" not in "" and r["final_script"].startswith("INT.")

# 2 loop -> regenerate -> critic pass
r = run("loop then recover", {D: [LOOP, GOOD], C: [PASS]}, [D, D, C], 2)

# 3 critic catches right/left arm -> targeted revision -> pass
r = run("critic catches continuity error", {D: [RIGHT_ARM, GOOD], C: [ARM, PASS]}, [D, C, D, C], 2)
assert "left sleeve" in r["final_script"]

# 4 revision is worse than original -> original kept
r = run("bad revisions never replace a better draft", {D: [RIGHT_ARM, LOOP, LOOP], C: [ARM]}, [D, C, D, D], 1)
assert "right sleeve" in r["final_script"]

# 5 critic API failure (both critic models down) -> fail-open, draft still returned
err = groq.APIConnectionError(request=httpx.Request("POST", "http://x"))
rag = make({D: [GOOD, err], C: [err]})          # fallback critic model == drafter model; give it an error too
r = rag.ask_agent(PROMPT, k_chunks=1)
print("PASS" if r["final_script"].startswith("INT.") and any(e.get("skipped") for e in r["trace"]) else "FAIL",
      "critic outage is fail-open | calls:", [c.split('/')[-1] for c in rag.client.calls])

# 6 hallucinated critic evidence is ignored
HALLU = '{"score": 3, "violations": [{"type":"continuity","severity":"major","evidence":"zebras ride unicycles through paris","problem":"x","fix":"y"}]}'
run("ungrounded critic objection ignored", {D: [GOOD], C: [HALLU]}, [D, C], 1)

# 7 INSUFFICIENT_CONTEXT -> retry retrieval once (legacy behaviour)
run("insufficient context retry", {D: ["INSUFFICIENT_CONTEXT", GOOD], C: [PASS]}, [D, D, C], 2)

# 8 legacy mode == one raw call, no repair, no critic
r = run("legacy (agentic=False)", {D: [BAD_BLANKS]}, [D], None, agentic=False)
assert r["final_script"] == BAD_BLANKS

# 9 truncated draft -> flagged, regenerated
run("truncated output regenerated", {D: [(GOOD, "length"), GOOD], C: [PASS]}, [D, D, C], 2)

# 10 deterministic repair of illegal blank lines + lowercase cue
reg = build_registry({"characters.md": open("sample_bible_for_tests/characters.md").read()})
rag = make({D: []})
fixed, repairs, loop = repair_script(BAD_BLANKS, rag.known_names)
rep = validate_script(fixed, user_prompt=PROMPT, registry=reg)
print("PASS" if "LENA\n(fast, bright)\nEspresso." in fixed and "ARTHUR\nMachine's" in fixed and not rep["errors"] else "FAIL", "blank-line repair |", repairs, rep["errors"], rep["warns"])
print("validator on RAW bad-blank draft (warns):", [w for w in validate_script(BAD_BLANKS, user_prompt=PROMPT, registry=reg)["warns"] if "broken" in w])