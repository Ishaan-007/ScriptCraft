"""
A/B benchmark: legacy single-shot  vs  agentic pipeline, on the same prompts.
Measures what the *user would actually see*:
  - rule-failure rate (loops, bad format, missing characters, too short/long, broken dialogue blocks)
  - independent judge score (a model different from the critic that guided the agent)
  - latency and number of LLM calls is visible in the trace
Usage:  GROQ_API_KEY=... python benchmark_agent.py --reps 3 --sleep 25
Free-tier note: gpt-oss-20b has 8K tokens/min and 200K tokens/day; keep --sleep high or add a payment method.
"""
import os, time, argparse, statistics as st
import pandas as pd
from rag_engine import ScriptCraftRAG, AgentConfig
from script_tools import validate_script

PROMPTS = [
    "Lena walks in Arthur's coffee shop to secretly investigate him. She orders an espresso and makes small talk, hinting at the secret about his arm, so that Arthur somehow reveals it.",
    "Silas Vane walks into The Rusted Mug to politely threaten Arthur about the Aegis Syndicate.",
    "Arthur closes the shop late at night and finds Lena waiting outside with a folder of evidence.",
    "A tense standoff at The Rusted Mug between Silas Vane, Arthur and Lena over a stolen data drive.",
    "Lena tries to bribe Arthur for his story while the espresso machine breaks down again.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--sleep", type=float, default=25.0, help="seconds between runs (rate limits)")
ap.add_argument("--judge", default=os.getenv("SCRIPTCRAFT_JUDGE_MODEL", "llama-3.3-70b-versatile"))
args = ap.parse_args()

rag = ScriptCraftRAG(os.environ["GROQ_API_KEY"], AgentConfig())
rows = []
for p in PROMPTS:
    for rep in range(args.reps):
        for mode in ("legacy", "agentic"):
            t0 = time.time()
            r = rag.ask_agent(p, k_chunks=3, agentic=(mode == "agentic"))
            dt = time.time() - t0
            text = r["final_script"]
            rep_ = validate_script(text, user_prompt=p, registry=rag.registry,
                                   min_words=rag.cfg.min_words, max_words=rag.cfg.max_words)
            j = rag.critique(p, r.get("context", ""), text, models=[args.judge, rag.cfg.critic_fallback_model])
            rows.append({"mode": mode, "prompt": p[:50], "rep": rep, "seconds": round(dt, 1),
                         "n_errors": len(rep_["errors"]), "errors": ";".join(rep_["errors"]),
                         "broken_blocks": any("broken_dialogue" in w for w in rep_["warns"]),
                         "judge_score": j["score"] if j else None,
                         "judge_major": len(j["major"]) if j else None,
                         "llm_calls": sum(1 for e in r["trace"] if e["step"] in ("draft", "critic")),
                         "words": rep_["stats"]["words"]})
            print(rows[-1]); time.sleep(args.sleep)

df = pd.DataFrame(rows); df.to_csv("benchmark_results.csv", index=False)
g = df.groupby("mode")
print("\n=== SUMMARY (lower errors / higher judge score is better) ===")
print(pd.DataFrame({
    "runs": g.size(),
    "rule_failure_rate": g["n_errors"].apply(lambda s: (s > 0).mean()),
    "broken_dialogue_rate": g["broken_blocks"].mean(),
    "judge_score_mean": g["judge_score"].mean(),
    "judge_major_issues": g["judge_major"].mean(),
    "seconds_mean": g["seconds"].mean(),
    "llm_calls_mean": g["llm_calls"].mean(),
}).round(3))