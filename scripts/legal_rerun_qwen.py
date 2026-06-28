# ============================================================
# STANDALONE CELL — Legal-only rerun for Qwen2.5-7B
# Fixes F: instead of hardcoded 0.5, the model answers each
# LegalBench Yes/No question itself; F = 1.0 if it matches the
# REAL ground-truth answer, 0.0 if not. This makes Legal a
# complete VERITAS-T test (F+C+S+M), not a reduced one.
#
# Run this in a FRESH Kaggle session (own GPU, no Qwen/Llama
# already loaded). Output: legal_rescored_qwen2.5-7b.csv
# Merge instructions printed at the end.
# ============================================================

!pip install -q transformers torch sentence-transformers tqdm

import re, json, math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_BASE = Path("/kaggle/input/datasets/samuelstephen77/veritas-datasets")  # adjust if needed
OUTPUT_DIR   = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"   # same model as the original Qwen run
EMBED_MODEL = "all-MiniLM-L6-v2"
K           = 50
GEN_MAX_NEW_TOKENS = 64
GEN_TEMPERATURE    = 0.7
SEED = 42

C_LEGAL    = 0.8
S_HIGH     = 0.8
S_LOW      = 0.2
LEGAL_HIGH_KEYWORDS = ["statute","section","court","case","ruling","penalty","liability",
                        "judgment","convicted","breach","regulation","compliance","infringement"]

np.random.seed(SEED)

# ── LOAD LEGAL DATA (same 400-row sample as the original run, same seed) ────
def load_legal_raw():
    dfs = []
    for fname in ["legal_train.tsv", "legal_test.tsv"]:
        p = DATASET_BASE / fname
        df = pd.read_csv(p, sep="\t", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df

legal_df = load_legal_raw()
print(f"Legal claims loaded: {len(legal_df)} (train+test combined; this is the FULL pool, "
      f"matches what the original run used since MAX_PER_DOMAIN=1500 > 400, no sampling occurred)")

# severity (unchanged logic from main pipeline)
def map_S(row):
    combined = f"{row.get('question','')} {row.get('contract','')}".lower()
    return S_HIGH if any(kw in combined for kw in LEGAL_HIGH_KEYWORDS) else S_LOW

legal_df["S"] = legal_df.apply(map_S, axis=1)
legal_df["C"] = C_LEGAL
legal_df["domain"] = "Legal"
legal_df["raw_text"] = legal_df["question"]
legal_df["ground_truth_answer"] = legal_df["answer"].str.strip()

# ── MODEL ────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
encoder = SentenceTransformer(EMBED_MODEL)

def answer_yes_no(question, contract_snippet):
    prompt = (f"Based on this contract excerpt, answer the question with exactly one word: Yes or No.\n\n"
              f"Contract: \"{str(contract_snippet)[:800]}\"\n\n"
              f"Question: {question}\nAnswer:")
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    if "yes" in text: return "Yes"
    if "no" in text: return "No"
    return "Unclear"

def sample_responses(prompt, k=K, temperature=GEN_TEMPERATURE):
    inputs = tok([str(prompt)[:512]] * k, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                              temperature=temperature, pad_token_id=tok.eos_token_id)
    return [tok.decode(out[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True) for i in range(k)]

# ── RUN: real F (generate-and-check) + real M (variance) per Legal claim ────
results = []
CHECKPOINT_EVERY = 50
ckpt_path = OUTPUT_DIR / "legal_rescored_qwen2.5-7b_checkpoint.csv"

start_idx = 0
if ckpt_path.exists():
    done_df = pd.read_csv(ckpt_path)
    start_idx = len(done_df)
    results = done_df.to_dict("records")
    print(f"Resuming from checkpoint: {start_idx} Legal claims already done.")

for i in tqdm(range(start_idx, len(legal_df)), desc="Legal rescoring [qwen2.5-7b]"):
    row = legal_df.iloc[i]
    model_answer = answer_yes_no(row["question"], row["contract"])
    F = 1.0 if model_answer == row["ground_truth_answer"] else 0.0

    responses = sample_responses(row["question"])
    embs = encoder.encode(responses, show_progress_bar=False)
    mu = embs.mean(axis=0)
    raw_var = float(np.mean(np.sum((embs - mu) ** 2, axis=1)))

    results.append({
        "domain": "Legal", "F": F, "C": row["C"], "S": row["S"],
        "raw_text": row["raw_text"], "ground_truth_answer": row["ground_truth_answer"],
        "model_answer": model_answer, "raw_variance": raw_var,
    })

    if (i + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(ckpt_path, index=False)
        print(f"  [checkpoint] {i+1}/{len(legal_df)} done")

out_df = pd.DataFrame(results)

# normalize raw_variance -> M (same min-max approach as main pipeline)
valid = out_df["raw_variance"].dropna()
vmin, vmax = valid.min(), valid.max()
out_df["M"] = out_df["raw_variance"].apply(
    lambda v: 0.5 if (pd.isna(v) or vmax == vmin) else 1.0 - (v - vmin) / (vmax - vmin)
)

final_path = OUTPUT_DIR / "legal_rescored_qwen2.5-7b.csv"
out_df.to_csv(final_path, index=False)

print("\n" + "=" * 70)
print("DONE.")
print(f"Saved: {final_path}")
print(f"F distribution: {out_df['F'].value_counts().to_dict()}  (mean F = {out_df['F'].mean():.3f})")
print(f"M mean: {out_df['M'].mean():.3f}, std: {out_df['M'].std():.3f}")
print("\nTO MERGE WITH YOUR EXISTING all_scored_qwen2.5-7b.csv:")
print("  1. Download this file: legal_rescored_qwen2.5-7b.csv")
print("  2. Bring it back along with all_scored_qwen2.5-7b.csv — Claude will merge")
print("     (replace the 400 Legal rows' F and M with these real values, keep S/C as-is).")
print("=" * 70)
