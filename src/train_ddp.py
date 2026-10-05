"""train_ddp.py — QLoRA SFT for HateMirage, launched via torchrun (DDP, 2xT4).
Launched by the Kaggle notebook as:
  torchrun --nproc_per_node=2 --master_port=29511 src/train_ddp.py --config out/train_config.json
Single-GPU fallback: python src/train_ddp.py --config ... (works, slower).
Env: HF_TOKEN must be set (Kaggle Secrets). Expects train/val CSVs with
  id,comment,target,intent,implication (+ optional evidence).
Pushes adapters to Hub every `push_every` steps if --hub_id given.
Timeout guard: --max_runtime_h (default 8.2) -> saves + exits 42 so notebook jumps to eval.
"""
import argparse, json, os, sys, time, math
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

PROMPT_VERSION = "v2-official-format"  # canonical text: prompts/parts.json (organizers' Zero-Shot.py/RAG.py format)
SYS_INTRO = "You are an expert in analyzing hateful comments driven by fake narratives.\n\nBased on the given comment, provide the requested analysis."
RAG_INTRO_TPL = "You are an expert in analyzing hateful comments driven by fake narratives.\n\nThe following context provides background information retrieved from external sources:\n[Context]: \"{context}\"\n\nBased on the context and the comment, provide the requested analysis."
TASK_TARGET = "If there is one target, only mention that. If there are multiple, mention each target as a single word and separate them by commas."
TASK_INTENT = "Briefly describe the *Intent* in a single concise sentence."
TASK_IMPLICATION = "Briefly describe the possible *Implication* in a single concise sentence."

def _intro(ev):
    return RAG_INTRO_TPL.replace("{context}", ev[:1200]) if ev else SYS_INTRO

def _norm_intent(s):
    s = str(s or "").strip()
    if s[:3].lower() == "to ":
        s = s[3:].lstrip()
    return "To " + s

def to_prompt(row, with_rag: bool):
    ev_raw = str(row.get("evidence", "")).strip() if with_rag else ""
    out = {}
    for task, label in (("TARGET", "Target"), ("INTENT", "Intent"), ("IMPLICATION", "Implication")):
        task_txt = {"TARGET": TASK_TARGET, "INTENT": TASK_INTENT, "IMPLICATION": TASK_IMPLICATION}[task]
        prompt = f"{_intro(ev_raw)}\n## Comment: \"{row['comment']}\"\n## Task: {task_txt}\n## {label}:"
        out[task.lower()] = prompt + " "
    out["target"] += f"{str(row['target']).strip()} </s>"
    out["intent"] += f"{_norm_intent(row['intent'])} </s>"
    out["implication"] += f"{str(row['implication']).strip()} </s>"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    t0 = time.time()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # each DDP rank loads its replica on its own GPU
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main: print(f"[train_ddp] config={cfg} world={os.environ.get('WORLD_SIZE','1')}", flush=True)

    train_df = pd.read_csv(cfg["train_csv"])
    val_df = pd.read_csv(cfg["val_csv"])
    with_rag = cfg.get("with_rag", True)
    rows = []
    for _, r in train_df.iterrows():
        for k, v in to_prompt(r, with_rag).items():
            rows.append({"text": v})
    train_ds = Dataset.from_list(rows)
    vrows = []
    for _, r in val_df.iterrows():
        for k, v in to_prompt(r, with_rag).items():
            vrows.append({"text": v})
    val_ds = Dataset.from_list(vrows)

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], token=(os.environ.get("HF_TOKEN") or None), trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
    _load_kw = dict(quantization_config=bnb,
        token=(os.environ.get("HF_TOKEN") or None), trust_remote_code=True,
        attn_implementation="eager")
    from transformers import AutoConfig as _AC
    _mcfg = _AC.from_pretrained(cfg["model_id"], token=(os.environ.get("HF_TOKEN") or None),
                                trust_remote_code=True)
    _mcfg.torch_dtype = torch.float16  # checkpoint may default to bf16 (T4-hostile)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], config=_mcfg,
                                                     dtype=torch.float16, **_load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], config=_mcfg,
                                                     torch_dtype=torch.float16, **_load_kw)
        if is_main: print("[train_ddp] from_pretrained fallback (torch_dtype)", flush=True)
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except TypeError:
        model = prepare_model_for_kbit_training(model)
        if is_main: print("[train_ddp] kbit_training fallback (no gradient_checkpointing kw)", flush=True)
    model.config.use_cache = False
    _n_bf16 = 0
    for _n, _p in model.named_parameters():
        if _p.requires_grad and _p.dtype == torch.bfloat16:
            _p.data = _p.data.to(torch.float32); _n_bf16 += 1
    if is_main: print(f"[train_ddp] cast {_n_bf16} trainable bf16 params to fp32", flush=True)
    peft = LoraConfig(r=cfg.get("lora_r", 64), lora_alpha=cfg.get("lora_alpha", 16), lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    # Warm-start: load adapter WEIGHTS only (fresh optimizer). Optimizer-state resume
    # crashes bitsandbytes kernels, so resume_ckpt (full state) is intentionally unsupported.
    from peft import PeftModel as _PM
    _warm_dir = cfg.get("warm_adapter") or None
    _warm_ok = False
    if _warm_dir:
        import glob as _gg
        _ad = _gg.glob(os.path.join(_warm_dir, "adapter_model.safetensors"))
        if _ad:
            try:
                model = _PM.from_pretrained(model, _warm_dir)
                _warm_ok = True
                if is_main: print(f"[train_ddp] warm-started adapter weights from {_warm_dir}", flush=True)
            except Exception as e:
                print(f"[train_ddp] warm load failed ({e}), random init", flush=True)
        else:
            print(f"[train_ddp] warm dir lacks adapter: {_warm_dir}", flush=True)
    import inspect as _insp
    _ta_params = set(_insp.signature(TrainingArguments.__init__).parameters)
    _strat_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
    _world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    _est_steps = max(1, (len(train_ds) * cfg.get("epochs", 3)) // max(1, (cfg.get("bs", 2) * _world * cfg.get("grad_accum", 8))))
    _ev = int(min(cfg.get("eval_steps", 200), max(5, _est_steps)))
    _do_best = _ev < _est_steps  # need >=1 eval before end for best-model + early-stop
    if is_main: print(f"[train_ddp] est_steps={_est_steps} eval_every={_ev} load_best={_do_best}", flush=True)
    _fingerprint = {"model_id": cfg["model_id"], "lora_r": cfg.get("lora_r", 64),
                    "n_train": len(train_ds), "epochs": cfg.get("epochs", 3),
                    "eff_batch": cfg.get("bs", 2) * _world * cfg.get("grad_accum", 8),
                    "max_steps": _est_steps}
    os.makedirs(cfg["output_dir"], exist_ok=True)
    json.dump(_fingerprint, open(os.path.join(cfg["output_dir"], "RESUME_INFO.json"), "w"), indent=1)
    _ta_want = dict(
        output_dir=cfg["output_dir"], num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("bs", 2), per_device_eval_batch_size=2,
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        optim="paged_adamw_32bit", fp16=False, bf16=False,
        gradient_checkpointing=True, logging_steps=25, eval_steps=_ev,
        save_steps=_ev, **{_strat_key: "steps"}, save_strategy="steps",
        save_total_limit=3, load_best_model_at_end=_do_best, metric_for_best_model="eval_loss",
        greater_is_better=False, ddp_find_unused_parameters=False,
        # NOTE: Hub push is notebook-side (manual, guarded). Trainer-side push is OFF:
        # SFTTrainer.__init__ calls create_repo and 403-crashes when the token lacks
        # write rights on the namespace. Training must never depend on Hub writes.
        push_to_hub=False, report_to="none", seed=42,
        max_grad_norm=0.3, weight_decay=0.01,
    )
    _dropped = sorted(k for k in _ta_want if k not in _ta_params)
    _ta_want = {k: v for k, v in _ta_want.items() if k in _ta_params}
    if is_main: print(f"[train_ddp] dropped TrainingArguments keys: {_dropped}", flush=True)
    args = TrainingArguments(**_ta_want)
    # timeout callback (holder: TrainerState has no save_model; use the trainer)
    from transformers import TrainerCallback
    _holder = {}
    class TimeoutCB(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            hrs = (time.time() - t0) / 3600
            if hrs > cfg.get("max_runtime_h", 8.2):
                print(f"[train_ddp] TIMEOUT {hrs:.2f}h -> save+exit42", flush=True)
                tr = _holder.get("trainer")
                if tr is not None:
                    try:
                        tr.save_model(cfg["output_dir"])
                    except Exception as e:
                        print("[train_ddp] timeout save failed:", e, flush=True)
                sys.stdout.flush(); os._exit(42)
    _sft_params = set(_insp.signature(SFTTrainer.__init__).parameters)
    _sft_kw = dict(args=args)
    if not _warm_ok:
        _sft_kw["peft_config"] = peft
    if "dataset_text_field" in _sft_params:
        _sft_kw["dataset_text_field"] = "text"
    if "tokenizer" in _sft_params:
        _sft_kw["tokenizer"] = tok
    elif "processing_class" in _sft_params:
        _sft_kw["processing_class"] = tok
    if "max_seq_length" in _sft_params:
        _sft_kw["max_seq_length"] = cfg.get("max_seq", 1536)
    elif "max_length" in _sft_params:
        _sft_kw["max_length"] = cfg.get("max_seq", 1536)
    if "packing" in _sft_params:
        _sft_kw["packing"] = False
    _cbs = [TimeoutCB()]
    if _do_best:
        _cbs.append(EarlyStoppingCallback(early_stopping_patience=2))
    _sft_kw["callbacks"] = _cbs
    os.makedirs(cfg["output_dir"], exist_ok=True)
    try:
        trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, **_sft_kw)
        _holder["trainer"] = trainer
    except Exception as e:
        import traceback as _tb
        open(os.path.join(cfg["output_dir"], "TRAIN_ERROR.txt"), "w").write(
            f"SFTTrainer build failed: {e}\nkwargs={sorted(_sft_kw)}\n" + _tb.format_exc())
        print("[train_ddp] SFTTrainer build failed, wrote TRAIN_ERROR.txt", flush=True)
        raise
    _n_cast = 0
    for _n, _p in trainer.model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.float32:
            _p.data = _p.data.to(torch.float32); _n_cast += 1
    if is_main: print(f"[train_ddp] post-peft cast {_n_cast} trainable params to fp32", flush=True)
    _resume = None  # full-state resume disabled (bnb optimizer-state crash); warm-start above instead
    if cfg.get("resume_ckpt"):
        print(f"[train_ddp] ignoring resume_ckpt (warm-start used): {cfg.get('resume_ckpt')}", flush=True)
    _fatal = None
    _fatal_tb = ""
    try:
        trainer.train(resume_from_checkpoint=_resume)
    except SystemExit:
        raise
    except Exception as e:
        import traceback as _tbr
        if _resume:
            print(f"[train_ddp] train-with-resume failed ({type(e).__name__}: {e}), retrying FRESH", flush=True)
            try:
                open(os.path.join(cfg["output_dir"], "TRAIN_ERROR.txt"), "a").write(
                    f"\nresume attempt failed, fresh retry: {e}\n" + _tbr.format_exc(limit=6))
            except Exception:
                pass
            _resume = None
            try:
                trainer.train(resume_from_checkpoint=None)
            except SystemExit:
                raise
            except Exception as e2:
                _fatal, _fatal_tb = e2, _tbr.format_exc()
        else:
            _fatal, _fatal_tb = e, _tbr.format_exc()
    if _fatal is not None:
        try:
            open(os.path.join(cfg["output_dir"], "TRAIN_ERROR.txt"), "a").write(
                f"\ntrainer.train failed: {_fatal}\n{_fatal_tb}")
        except Exception:
            pass
        print("[train_ddp] ERROR, saving partial:", _fatal, flush=True)
        trainer.save_model(cfg["output_dir"] + "-partial")
        raise _fatal
    trainer.save_model(cfg["output_dir"])
    if is_main:
        tok.save_pretrained(cfg["output_dir"])
        hrs = (time.time() - t0) / 3600
        json.dump({"hours": hrs, "hub": cfg.get("hub_id")}, open(os.path.join(cfg["output_dir"], "train_done.json"), "w"), indent=2)
        print(f"[train_ddp] DONE {hrs:.2f}h", flush=True)

if __name__ == "__main__":
    main()
