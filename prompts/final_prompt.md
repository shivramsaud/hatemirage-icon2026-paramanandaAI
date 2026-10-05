# FINAL PROMPT — HateMirage (v2-official-format)

Single source of truth: `parts.json` (machine). Injected verbatim into training and inference.
Official source: https://github.com/Sai-Kartheek-Reddy/HateMirage/blob/main/code/README.md
("Prompt Template for Explanation Generation").

## Runtime format (per field, greedy T=0)
```
{intro or RAG-intro with [Context]}
## Comment: "{comment}"
## Task: {field instruction}
## {Target|Intent|Implication}:
```
RAG: top-5 mpnet/FAISS cosine over the official `RAG_Reference_Data.jsonl`, concatenated.
Post: strip, collapse whitespace, first 2 sentences, enforce `To…` (Intent) / `Could…` (Implication).
Eval: MiniLM-L6-v2 cosine + ROUGE-L F1 with stemmer (organizers' scorer).
