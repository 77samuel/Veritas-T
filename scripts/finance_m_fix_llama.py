# ============================================================
# STANDALONE CELL — Fix 196 Finance claims with missing M (Llama)
#
# CAUSE: Kaggle session was cut off mid-run at ~claim 2200/2396.
# On resume, the last 196 Finance claims got processed but their
# M values were never written correctly to the final output due
# to a bug in the checkpoint-resume normalization step. This
# script recomputes M for ONLY those 196 claims and saves a
# small patch file to merge back into all_scored_llama3-8b.csv.
#
# Run in a FRESH Kaggle session (GPU attached, dataset attached).
# Output: finance_m_fix_llama3-8b.csv
# ============================================================

!pip install -q transformers torch sentence-transformers tqdm

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_BASE = Path("/kaggle/input/datasets/samuelstephen77/veritas-datasets")
OUTPUT_DIR   = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
K           = 50
GEN_MAX_NEW_TOKENS = 64
GEN_TEMPERATURE    = 0.7
SUB_BATCH   = 20   # same value that worked for the main Llama run

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded.")
except Exception as e:
    HF_TOKEN = None
    print(f"[WARN] No HF_TOKEN ({e}). Llama-3 will fail without it.")

# ── UPLOAD YOUR EXISTING all_scored_llama3-8b.csv AS A KAGGLE DATASET FIRST,
#    OR PASTE THE RAW_TEXT LIST BELOW IF EASIER. This version re-derives
#    Finance and finds the same 196 rows by re-running load_finance() with
#    the same seed/config, so it doesn't need the broken file as input.
# ────────────────────────────────────────────────────────────────────────────
import re
from datetime import datetime

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"({MONTHS})\s+\d{{1,2}},?\s+(\d{{4}})")
FINANCE_HIGH_KEYWORDS = ["revenue","earnings","eps","profit","loss","debt","liability",
                          "dividend","share price","guidance","forecast","operating income",
                          "net income","cash flow","equity","assets","turnover"]
SEED = 42
C_FINANCE = 0.9
S_HIGH, S_LOW = 0.8, 0.2

def extract_finance_t(context_text, reference_date=datetime(2024, 6, 30)):
    matches = DATE_RE.findall(str(context_text))
    if not matches: return None
    years = [int(y) for _, y in matches]
    most_recent_year = max(years)
    months_map = {m: i+1 for i, m in enumerate(MONTHS.split("|"))}
    candidate_months = [months_map[m] for m, y in matches if int(y) == most_recent_year]
    month = max(candidate_months) if candidate_months else 6
    try:
        filing_date = datetime(most_recent_year, month, 1)
    except ValueError:
        return None
    age = (reference_date.year - filing_date.year) * 12 + (reference_date.month - filing_date.month)
    return max(age, 0)

def load_finance_full():
    df = pd.read_csv(DATASET_BASE / "finance_data.csv", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ground_truth_label" in df.columns:
        df["F"] = df["ground_truth_label"].apply(
            lambda x: 0.0 if "hallucination" in str(x).lower() and "not" not in str(x).lower() else 1.0)
    else:
        df["F"] = 0.5
    text_cols = [c for c in df.columns if any(k in c for k in ["question","query","text","input","context","passage"])]
    def map_S(row):
        combined = " ".join(str(row.get(c,"")) for c in text_cols).lower()
        return S_HIGH if any(kw in combined for kw in FINANCE_HIGH_KEYWORDS) else S_LOW
    df["S"] = df.apply(map_S, axis=1)
    df["C"] = C_FINANCE
    df["domain"] = "Finance"
    df["raw_text"] = df["query"] if "query" in df.columns else df.index.astype(str)
    df["t_real"] = df["context"].apply(extract_finance_t) if "context" in df.columns else None
    # NOTE: original run used MAX_PER_DOMAIN=1500 > 896, so NO sampling occurred.
    # This means the full 896-row Finance set, in this exact order, matches the
    # original run exactly (same as we verified for Legal via S/C alignment).
    return df[["F","C","S","raw_text","domain","t_real"]].dropna(subset=["F"]).reset_index(drop=True)

finance_full = load_finance_full()
print(f"Finance claims re-derived: {len(finance_full)} (should be 896, matching original run)")

# The broken rows were positions 700-895 (last 196 of the 896 Finance claims)
MISSING_START = 700
missing_df = finance_full.iloc[MISSING_START:].reset_index(drop=True)
print(f"Targeting {len(missing_df)} claims (positions {MISSING_START}-895) for M recomputation.")

# ── MODEL ────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)
encoder = SentenceTransformer(EMBED_MODEL)

def sample_responses(prompt, k=K, temperature=GEN_TEMPERATURE, sub_batch=SUB_BATCH):
    responses = []
    for start in range(0, k, sub_batch):
        n = min(sub_batch, k - start)
        inputs = tok([str(prompt)[:512]] * n, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                                  temperature=temperature, pad_token_id=tok.eos_token_id)
        for i in range(n):
            responses.append(tok.decode(out[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        del inputs, out
        torch.cuda.empty_cache()
    return responses

# ── COMPUTE RAW VARIANCE FOR THE 196 MISSING CLAIMS ──────────────────────────
raw_vars = []
ckpt_path = OUTPUT_DIR / "finance_m_fix_checkpoint.csv"
start_idx = 0
if ckpt_path.exists():
    done_df = pd.read_csv(ckpt_path)
    start_idx = len(done_df)
    raw_vars = done_df["raw_variance"].tolist()
    print(f"Resuming from checkpoint: {start_idx} already done.")

for i in tqdm(range(start_idx, len(missing_df)), desc="Fixing Finance M [llama3-8b]"):
    text = missing_df.iloc[i]["raw_text"]
    responses = sample_responses(text)
    embs = encoder.encode(responses, show_progress_bar=False)
    mu = embs.mean(axis=0)
    raw_var = float(np.mean(np.sum((embs - mu) ** 2, axis=1)))
    raw_vars.append(raw_var)

    if (i + 1) % 50 == 0:
        pd.DataFrame({"raw_variance": raw_vars}).to_csv(ckpt_path, index=False)
        print(f"  [checkpoint] {i+1}/{len(missing_df)} done")

missing_df = missing_df.iloc[:len(raw_vars)].copy()
missing_df["raw_variance"] = raw_vars

# normalize using the SAME min-max approach, but anchored to the full Finance
# distribution's plausible range (use this batch's own min/max since the
# original batch's raw variances were not saved)
vmin, vmax = min(raw_vars), max(raw_vars)
missing_df["M"] = missing_df["raw_variance"].apply(
    lambda v: 0.5 if vmax == vmin else 1.0 - (v - vmin) / (vmax - vmin)
)

final_path = OUTPUT_DIR / "finance_m_fix_llama3-8b.csv"
missing_df.to_csv(final_path, index=False)

print("\n" + "=" * 70)
print("DONE.")
print(f"Saved: {final_path}")
print(f"M mean: {missing_df['M'].mean():.3f}, std: {missing_df['M'].std():.3f}")
print("\nBring this file back along with all_scored_llama3-8b.csv.")
print("Claude will merge these 196 rows into the 196 NaN positions.")
print("=" * 70)
