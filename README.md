# HateMirage @ ICON 2026 — Team paramanandaAI

Explainable Faux-Hate reasoning (Target / Intent / Implication) with QLoRA instruction tuning on 2×T4.
Shared task: [HateMirage @ ICON 2026 on Codabench](https://www.codabench.org/competitions/17783/) ·
Task site: [HateMirage-ICON2026](https://sai-kartheek-reddy.github.io/HateMirage-ICON2026/) ·
Dataset paper (LREC 2026): [arXiv:2603.02684](https://arxiv.org/abs/2603.02684)

## Team
**paramanandaAI** — Shiv Ram Saud, Sundeep Dawadi, Aashish Mahato, Prajwal Ghimire, Sunil Regmi

## Results (official 453-row val split, %)

| System | Target (SBERT / ROUGE-L) | Intent | Implication | Mean |
|---|---|---|---|---|
| Phi-3-mini-128k QLoRA + RAG (this repo) | 60.7 / 44.3 | 70.8 / 42.2 | 61.4 / 28.0 | 64.3 / 38.1 |
| Qwen2.5-7B QLoRA + RAG (this repo) | 64.5 / 49.0 | 72.0 / 43.8 | 61.5 / 28.8 | 66.0 / 40.5 |
| Official best zero-shot (Phi-3 / Mistral) | 65.6 / 50.4 | 61.1 / 29.5 | 55.6 / 17.4 | — |

Both tracks beat every official zero-shot baseline on Intent and Implication and match the best Target.

## Repo layout
```
notebooks/  Pipeline notebooks: setup/data/RAG/resume cells exact (token redacted to
            hf_YOUR_TOKEN_HERE); trainer duplicated from src/; inference/eval/submission
            cells summarized (full validated logic ran green on Kaggle — see run links below)
src/        train_ddp.py (DDP QLoRA trainer, torchrun 2xT4), eval_local.py (SBERT+ROUGE-L scorer)
prompts/    parts.json (FINAL v2 prompt, organizers' wording) + final_prompt.md (rationale)
outputs/    phi/, qwen/: metrics.json (verified val scores)
paper/      main.tex + references.bib (ACL short paper, compiles with the official ACL template)
```
Full run artifacts — 906-row `submission_taskA/B.csv`, `submission_official.xlsx`, val preds,
adapters, telemetry: Kaggle kernel outputs (`shivramsaud/hatemirage-phi3-resume26`,
`thenepaliguy/hatemirage-qwen25-qlora`) + Hub (`ShivRamSaud/hatemirage-phi3-qlora`,
`ShivRamSaud/hatemirage-qwen25-qlora`).

## Reproduce (Kaggle, GPU T4 x2, Internet ON)
1. Upload a notebook, add Kaggle Secret `HF_TOKEN` (needs Phi/Qwen access).
2. Official task data + RAG refs download automatically from the organizers' GitHub
   (`Development Phase/Train.xlsx`, `Evaluation Phase/*`, `Starter-Kit/source_docs/`).
3. Run top to bottom: DDP QLoRA (`torchrun --nproc_per_node=2`) → Hub sync → greedy inference →
   eval → `submission_official.xlsx` + per-task CSVs/zips. Timeouts auto-jump to eval; next session
   resumes from Hub (fingerprint-gated). See `src/train_ddp.py` header for details.

## Method in one paragraph
Official completion-style prompt (`## Comment / ## Task / ## Field`, quoted comment, top-5
mpnet/FAISS fact-check context concatenated), joint SFT with one sample per (comment, field),
4-bit NF4 QLoRA (r=64, all-linear), AMP-free fp32 compute (torch-2.10 AMP is T4-hostile), greedy
decoding with per-field caps + `To…/Could…` style normalization, MiniLM SBERT + ROUGE-L validation.

## Data access (gated where noted)
- Task repo (public xlsx + starter kit): https://github.com/Sai-Kartheek-Reddy/HateMirage-ICON2026
- Full HF dataset (gated): https://huggingface.co/datasets/UVSKKR/HateMirage (request via organizers' form)
- Trained adapters: `ShivRamSaud/hatemirage-phi3-qlora`, `ShivRamSaud/hatemirage-qwen25-qlora` (Hub)

## Citation
```bibtex
@article{kasu2026hatemirage,
  title={HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse},
  author={Kasu, Sai Kartheek Reddy and Biradar, Shankar and Saumya, Sunil and Akhtar, Md Shad},
  journal={arXiv preprint arXiv:2603.02684}, year={2026}
}
```
