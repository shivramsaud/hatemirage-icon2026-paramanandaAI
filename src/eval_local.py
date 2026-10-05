"""Local replica of HateMirage Codabench scoring: SBERT cosine + ROUGE-L F1.
Usage:
  python eval_local.py --pred preds_val.csv --ref val.csv
CSV contract (auto-detects): id | comment/text | target | intent | implication
Requires: sentence-transformers, rouge-score, pandas
"""
import argparse, json, re
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np

FIELDS = ["target", "intent", "implication"]

def norm(s: str) -> str:
    s = str(s or "").strip()
    return re.sub(r"\s+", " ", s)

def sbert_sim(model, a_list, b_list):
    ea = model.encode(a_list, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
    eb = model.encode(b_list, convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=True)
    return [float(np.dot(x, y)) for x, y in zip(ea, eb)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    ref = pd.read_csv(args.ref)
    pred.columns = [c.lower().strip() for c in pred.columns]
    ref.columns = [c.lower().strip() for c in ref.columns]
    # unify comment col
    for df in (pred, ref):
        if "comment" not in df.columns:
            for alt in ("text", "input", "sentence"):
                if alt in df.columns:
                    df.rename(columns={alt: "comment"}, inplace=True)
    key = "id" if "id" in pred.columns and "id" in ref.columns else None
    if key:
        ref = ref.set_index("id").loc[pred["id"]].reset_index()
    assert len(pred) == len(ref), f"row mismatch {len(pred)} vs {len(ref)}"

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model = SentenceTransformer(args.sbert)
    out, means = {}, {"sbert": [], "rougeL": []}
    for f in FIELDS:
        if f not in pred.columns or f not in ref.columns:
            print(f"[skip] {f} missing"); continue
        P = [norm(x) for x in pred[f].tolist()]
        R = [norm(x) for x in ref[f].tolist()]
        rl = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(R, P)]
        sb = sbert_sim(model, P, R)
        out[f] = {"sbert_mean": float(np.mean(sb)), "rougeL_mean": float(np.mean(rl)), "n": len(P)}
        means["sbert"].append(out[f]["sbert_mean"]); means["rougeL"].append(out[f]["rougeL"])
        print(f"{f:12s} SBERT {out[f]['sbert_mean']:.4f}  ROUGE-L {out[f]['rougeL_mean']:.4f}")
    out["mean"] = {"sbert": float(np.mean(means["sbert"])), "rougeL": float(np.mean(means["rougeL"])),
                   "combined": float((np.mean(means["sbert"]) + np.mean(means["rougeL"])) / 2)}
    print("MEAN:", json.dumps(out["mean"], indent=2))
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
