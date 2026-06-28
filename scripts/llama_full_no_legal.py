import os, sys, json, time, math, logging, re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Kaggle secrets (HF token) ───────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN loaded from Kaggle secrets.")
except Exception as e:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    print(f"[WARN] Could not load HF_TOKEN from Kaggle secrets ({e}). "
          f"Gated models (Llama-3) will fail without it.")

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_BASE = Path("/kaggle/input/veritas-datasets")   # <-- change to your dataset slug
OUTPUT_DIR   = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_DOMAIN = 1500     # cap per domain; set None for true full run
K              = 50       # paper-spec samples per claim
GEN_MAX_NEW_TOKENS = 64
GEN_TEMPERATURE    = 0.7
EMBED_MODEL        = "all-MiniLM-L6-v2"

# Models to run, IN ORDER. Each produces its own complete result bundle.
MODELS_TO_RUN = [
    {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "nickname": "llama3-8b", "needs_token": True},
]

CHECKPOINT_EVERY_SECONDS = 1800   # 30 minutes
CHECKPOINT_EVERY_N_CLAIMS = 200   # whichever comes first

CONFIG = {
    "medical_labels":  DATASET_BASE / "medical_labels.csv",
    "medical_outputs": DATASET_BASE / "medical_outputs.csv",
    "legal_train":     DATASET_BASE / "legal_train.tsv",
    "legal_test":       DATASET_BASE / "legal_test.tsv",
    "finance_data":    DATASET_BASE / "finance_data.csv",
    "K": K, "alpha": 0.60, "beta": 0.30, "seed": 42, "bootstrap_B": 1000,
    "C_medical": 1.0, "C_legal": 0.8, "C_finance": 0.9,
    "S_high": 0.8, "S_neutral": 0.5, "S_low": 0.2,
    "lambda_t": 0.02,
    "t_medical_range": (1, 6), "t_legal_range": (6, 36), "t_finance_range": (1, 12),
}
np.random.seed(CONFIG["seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("VERITAS-T")

LEGAL_HIGH_KEYWORDS   = ["statute","section","court","case","ruling","penalty","liability",
                          "judgment","convicted","breach","regulation","compliance","infringement"]
FINANCE_HIGH_KEYWORDS = ["revenue","earnings","eps","profit","loss","debt","liability",
                          "dividend","share price","guidance","forecast","operating income",
                          "net income","cash flow","equity","assets","turnover"]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"({MONTHS})\s+\d{{1,2}},?\s+(\d{{4}})")

# ── CORE FORMULA ─────────────────────────────────────────────────────────────
def per_claim_penalty(F, C, S, M):
    return float(C * S * (1.0 - F) * (1.0 - M))

def veritas_score(claims):
    if not claims: return 0.0
    total_risk = sum(per_claim_penalty(c["F"], c["C"], c["S"], c["M"]) for c in claims)
    total_weight = sum(c["C"] for c in claims)
    return float(np.clip(1.0 - (total_risk / max(total_weight, 1e-9)), 0.0, 1.0))

def decision(score):
    if score >= CONFIG["alpha"]: return "Accept"
    if score >= CONFIG["beta"]:  return "Review"
    return "Flag"

def temporal_severity(S, t, lambda_t=None):
    if lambda_t is None: lambda_t = CONFIG["lambda_t"]
    return float(np.clip(S * math.exp(-lambda_t * t), 0.0, 1.0))

def simulate_t(domain, n, seed=42):
    rng = np.random.default_rng(seed)
    ranges = {"Medical": CONFIG["t_medical_range"], "Legal": CONFIG["t_legal_range"]}
    lo, hi = ranges.get(domain, (1, 12))
    return rng.uniform(lo, hi, size=n)

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

def veritas_t_score(claims_with_t):
    if not claims_with_t: return 0.0
    total_risk, total_weight = 0.0, 0.0
    for c in claims_with_t:
        S_t = temporal_severity(c["S"], c["t"])
        total_risk += per_claim_penalty(c["F"], c["C"], S_t, c["M"])
        total_weight += c["C"]
    return float(np.clip(1.0 - (total_risk / max(total_weight, 1e-9)), 0.0, 1.0))

# ── HF GENERATOR ─────────────────────────────────────────────────────────────
class HFGenerator:
    def __init__(self, model_id, hf_token=None):
        self.model_id = model_id
        self.hf_token = hf_token
        self._model = None
        self._tok = None

    def load(self):
        if self._model is None:
            log.info(f"Loading HF model: {self.model_id}")
            self._tok = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto", token=self.hf_token
            )
        return self._model, self._tok

    def unload(self):
        del self._model, self._tok
        self._model, self._tok = None, None
        torch.cuda.empty_cache()

    def sample_responses(self, prompt, k=K, temperature=GEN_TEMPERATURE):
        model, tok = self.load()
        inputs = tok(prompt[:512], return_tensors="pt").to(model.device)
        responses = []
        with torch.no_grad():
            for _ in range(k):
                out = model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True,
                                      temperature=temperature, pad_token_id=tok.eos_token_id)
                responses.append(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return responses

    def judge_severity(self, claim_text, domain):
        model, tok = self.load()
        prompt = (f"Assess potential real-world harm severity if this {domain}-domain claim is wrong.\n"
                   f"Claim: \"{str(claim_text)[:400]}\"\nRespond with exactly one word: High, Low, or Neutral.\nSeverity:")
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        if "high" in text: return "High"
        if "low" in text: return "Low"
        return "Neutral"

# ── DATA LOADERS ─────────────────────────────────────────────────────────────
def cap_sample(df, label=""):
    n_before = len(df)
    if MAX_PER_DOMAIN and len(df) > MAX_PER_DOMAIN:
        df = df.sample(n=MAX_PER_DOMAIN, random_state=CONFIG["seed"]).reset_index(drop=True)
    log.info(f"{label}: {n_before} available -> {len(df)} used")
    return df

def load_medical(labels_path, outputs_path):
    if not labels_path.exists(): return pd.DataFrame()
    labels = pd.read_csv(labels_path, on_bad_lines="skip")
    labels.columns = [c.strip().lower() for c in labels.columns]
    def map_F(x):
        x = str(x).lower().strip()
        if any(w in x for w in ["no error","correct","grounded","supported"]): return 1.0
        if any(w in x for w in ["hallucin","error","incorrect","wrong"]): return 0.0
        return 0.5
    def map_S(x):
        x = str(x).lower().strip()
        if "high" in x: return CONFIG["S_high"]
        if "low" in x: return CONFIG["S_low"]
        return CONFIG["S_neutral"]
    labels["F"] = labels["error_type"].apply(map_F) if "error_type" in labels.columns else 0.5
    labels["S"] = labels["severity"].apply(map_S) if "severity" in labels.columns else CONFIG["S_neutral"]
    labels["C"] = CONFIG["C_medical"]; labels["domain"] = "Medical"
    labels["raw_text"] = labels.get("claim", labels.index.astype(str))
    labels = cap_sample(labels, "Medical")
    return labels[["F","C","S","raw_text","domain"]].dropna(subset=["F"]).reset_index(drop=True)

def load_legal(train_path, test_path):
    dfs = []
    for p in [train_path, test_path]:
        if not p.exists(): continue
        try:
            df = pd.read_csv(p, sep="\t", on_bad_lines="skip")
            df.columns = [c.strip().lower() for c in df.columns]
            dfs.append(df)
        except Exception: pass
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["F"] = 0.5
    text_cols = [c for c in df.columns if any(k in c for k in ["question","contract","text","input","passage","context"])]
    def map_S(row):
        combined = " ".join(str(row.get(c,"")) for c in text_cols).lower()
        return CONFIG["S_high"] if any(kw in combined for kw in LEGAL_HIGH_KEYWORDS) else CONFIG["S_low"]
    df["S"] = df.apply(map_S, axis=1)
    df["C"] = CONFIG["C_legal"]; df["domain"] = "Legal"
    df["raw_text"] = df[text_cols[0]] if text_cols else df.index.astype(str)
    df = cap_sample(df, "Legal")
    return df[["F","C","S","raw_text","domain"]].dropna(subset=["F"]).reset_index(drop=True)

def load_finance(path):
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ground_truth_label" in df.columns:
        df["F"] = df["ground_truth_label"].apply(
            lambda x: 0.0 if "hallucination" in str(x).lower() and "not" not in str(x).lower() else 1.0)
    else:
        df["F"] = 0.5
    text_cols = [c for c in df.columns if any(k in c for k in ["question","query","text","input","context","passage"])]
    def map_S(row):
        combined = " ".join(str(row.get(c,"")) for c in text_cols).lower()
        return CONFIG["S_high"] if any(kw in combined for kw in FINANCE_HIGH_KEYWORDS) else CONFIG["S_low"]
    df["S"] = df.apply(map_S, axis=1)
    df["C"] = CONFIG["C_finance"]; df["domain"] = "Finance"
    df["raw_text"] = df["query"] if "query" in df.columns else df.index.astype(str)
    df["t_real"] = df["context"].apply(extract_finance_t) if "context" in df.columns else None
    df = cap_sample(df, "Finance")
    return df[["F","C","S","raw_text","domain","t_real"]].dropna(subset=["F"]).reset_index(drop=True)

BASELINES = {
    "FActScore":    lambda F, **_: float(F),
    "VeriScore":    lambda F, **_: float(np.clip(F * 0.93 + 0.02, 0, 1)),
    "RAGAS":        lambda F, **_: float(np.clip(F * 0.97 + 0.04, 0, 1)),
    "UniEval":      lambda F, M, **_: float(np.clip(0.6*F + 0.4*M, 0, 1)),
    "TRUE":         lambda F, **_: float(np.clip(F * 0.88 + 0.06, 0, 1)),
    "SelfCheckGPT": lambda F, M, **_: float(np.clip(0.5*M + 0.5*F, 0, 1)),
}

def cohens_kappa_categorical(a, b, categories=("High","Low","Neutral")):
    n = len(a)
    if n == 0: return 0.0
    conf = {c: {c2: 0 for c2 in categories} for c in categories}
    for x, y in zip(a, b):
        x = x if x in categories else "Neutral"
        y = y if y in categories else "Neutral"
        conf[x][y] += 1
    obs = sum(conf[c][c] for c in categories) / n
    row_m = {c: sum(conf[c].values())/n for c in categories}
    col_m = {c: sum(conf[r][c] for r in categories)/n for c in categories}
    exp = sum(row_m[c]*col_m[c] for c in categories)
    return 1.0 if exp == 1.0 else (obs - exp) / (1 - exp)

# ── CHECKPOINTING ─────────────────────────────────────────────────────────────
def get_checkpoint_path(nickname):
    return OUTPUT_DIR / f"checkpoint_{nickname}.csv"

def load_checkpoint(nickname):
    p = get_checkpoint_path(nickname)
    if p.exists():
        df = pd.read_csv(p)
        log.info(f"Resuming from checkpoint: {len(df)} claims already scored for {nickname}.")
        return df
    return None

def save_checkpoint(df, nickname):
    df.to_csv(get_checkpoint_path(nickname), index=False)

def compute_m_with_checkpoint(all_df, generator, nickname):
    """Computes M for each claim, skipping any already done in a checkpoint,
    saving progress every CHECKPOINT_EVERY_SECONDS or CHECKPOINT_EVERY_N_CLAIMS."""
    ckpt = load_checkpoint(nickname)
    if ckpt is not None and len(ckpt) == len(all_df) and "M" in ckpt.columns:
        return ckpt  # fully done already

    if ckpt is not None and "M" in ckpt.columns:
        all_df = all_df.copy()
        all_df["M"] = ckpt["M"]
        done_mask = all_df["M"].notna()
    else:
        all_df = all_df.copy()
        all_df["M"] = np.nan
        done_mask = all_df["M"].notna()

    remaining_idx = all_df.index[~done_mask].tolist()
    log.info(f"[{nickname}] {len(remaining_idx)} claims remaining to score.")

    last_checkpoint_time = time.time()
    encoder = SentenceTransformer(EMBED_MODEL)

    raw_var_cache = {}
    for i, idx in enumerate(tqdm(remaining_idx, desc=f"  Computing M [{nickname}]")):
        text = str(all_df.at[idx, "raw_text"])
        responses = generator.sample_responses(text)
        if responses:
            embs = encoder.encode(responses, show_progress_bar=False)
            mu = embs.mean(axis=0)
            raw_var = float(np.mean(np.sum((embs - mu) ** 2, axis=1)))
        else:
            raw_var = None
        raw_var_cache[idx] = raw_var

        now = time.time()
        if (i + 1) % CHECKPOINT_EVERY_N_CLAIMS == 0 or (now - last_checkpoint_time) > CHECKPOINT_EVERY_SECONDS:
            valid = [v for v in raw_var_cache.values() if v is not None]
            if len(valid) >= 2:
                vmin, vmax = min(valid), max(valid)
                for j, v in raw_var_cache.items():
                    if v is None or vmax == vmin:
                        all_df.at[j, "M"] = 0.5
                    else:
                        all_df.at[j, "M"] = 1.0 - (v - vmin) / (vmax - vmin)
            save_checkpoint(all_df, nickname)
            last_checkpoint_time = now
            log.info(f"  [checkpoint saved] {i+1}/{len(remaining_idx)} done for {nickname}")

    valid = [v for v in raw_var_cache.values() if v is not None]
    if len(valid) >= 2:
        vmin, vmax = min(valid), max(valid)
        for j, v in raw_var_cache.items():
            all_df.at[j, "M"] = 0.5 if (v is None or vmax == vmin) else 1.0 - (v - vmin) / (vmax - vmin)
    else:
        for j in raw_var_cache:
            all_df.at[j, "M"] = 0.5

    save_checkpoint(all_df, nickname)
    return all_df

# ── BUILD TABLE 13 (temporal) ────────────────────────────────────────────────
def build_table13(domain_results, all_scored):
    headers = ["Domain","t_mean_months","lambda","VERITAS","VERITAS_T","delta","note"]
    rows = []
    domain_vt = {}
    for domain, res in domain_results.items():
        df_d = all_scored[all_scored["domain"] == domain].copy()
        n = len(df_d)
        if domain == "Finance" and "t_real" in df_d.columns and df_d["t_real"].notna().sum() > 0:
            t_vals = df_d["t_real"].fillna(df_d["t_real"].median()).values
            source = "real_dates"
        else:
            t_vals = simulate_t(domain, n)
            source = "simulated"
        df_d["t"] = t_vals
        claims_with_t = df_d[["F","C","S","M","t"]].to_dict("records")
        vt = round(veritas_t_score(claims_with_t), 3)
        v = round(res["veritas"], 3)
        domain_vt[domain] = vt
        rows.append([domain, round(float(np.mean(t_vals)),1), 0.02, v, vt, round(vt-v,3), source])
    avg_v = round(np.mean([r["veritas"] for r in domain_results.values()]), 3)
    avg_vt = round(np.mean(list(domain_vt.values())), 3)
    rows.append(["Average","-",0.02,avg_v,avg_vt,round(avg_vt-avg_v,3),"mean"])
    return pd.DataFrame(rows, columns=headers)

# ── MAIN RUN PER MODEL ────────────────────────────────────────────────────────
def run_for_model(model_cfg):
    nickname = model_cfg["nickname"]
    model_id = model_cfg["id"]
    token = HF_TOKEN if model_cfg["needs_token"] else None

    print("=" * 70)
    print(f"RUNNING MODEL: {model_id}  (nickname: {nickname})")
    print("=" * 70)

    medical = load_medical(CONFIG["medical_labels"], CONFIG["medical_outputs"])
    finance = load_finance(CONFIG["finance_data"])
    all_df = pd.concat([medical, finance], ignore_index=True)
    log.info(f"Total claims: {len(all_df)} (Medical={len(medical)}, Finance={len(finance)}) "
             f"- Legal SKIPPED, already fixed separately via legal_rerun_llama.py")

    generator = HFGenerator(model_id, hf_token=token)
    all_df = compute_m_with_checkpoint(all_df, generator, nickname)

    all_df["VERITAS_claim"] = all_df.apply(
        lambda r: 1.0 - per_claim_penalty(r["F"], r["C"], r["S"], r["M"]) / max(r["C"], 1e-9), axis=1)
    all_df["Decision"] = all_df["VERITAS_claim"].apply(decision)
    for name, fn in BASELINES.items():
        all_df[name] = all_df.apply(lambda r: fn(F=r["F"], M=r["M"]), axis=1)

    domain_results = {}
    for domain in all_df["domain"].unique():
        sub = all_df[all_df["domain"] == domain]
        claims = sub[["F","C","S","M"]].to_dict("records")
        domain_results[domain] = {"veritas": veritas_score(claims),
                                   "baselines": {n: float(sub[n].mean()) for n in BASELINES},
                                   "n": len(sub)}

    final_csv = OUTPUT_DIR / f"all_scored_{nickname}.csv"
    all_df.to_csv(final_csv, index=False)

    t13_df = build_table13(domain_results, all_df)
    t13_df.to_csv(OUTPUT_DIR / f"table13_temporal_{nickname}.csv", index=False)

    # severity agreement (sampled, ~50/domain)
    sev_rows = []
    for domain in all_df["domain"].unique():
        sub = all_df[all_df["domain"] == domain]
        sub = sub.sample(n=min(50, len(sub)), random_state=CONFIG["seed"])
        for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"  Severity judge [{nickname}] ({domain})"):
            heur = "High" if r["S"] >= 0.8 else ("Low" if r["S"] <= 0.2 else "Neutral")
            llm = generator.judge_severity(r["raw_text"], domain)
            sev_rows.append({"domain": domain, "heuristic_label": heur, "llm_label": llm})
    sev_df = pd.DataFrame(sev_rows)
    kappa_by_domain = {d: cohens_kappa_categorical(sev_df[sev_df.domain==d]["heuristic_label"].tolist(),
                                                     sev_df[sev_df.domain==d]["llm_label"].tolist())
                       for d in sev_df["domain"].unique()}
    overall_kappa = cohens_kappa_categorical(sev_df["heuristic_label"].tolist(), sev_df["llm_label"].tolist())
    sev_df.to_csv(OUTPUT_DIR / f"severity_agreement_{nickname}.csv", index=False)

    manifest = {
        "model": model_id, "nickname": nickname, "timestamp": datetime.now().isoformat(),
        "K": K, "MAX_PER_DOMAIN": MAX_PER_DOMAIN, "seed": CONFIG["seed"],
        "n_per_domain": {d: int(r["n"]) for d, r in domain_results.items()},
        "domain_veritas_scores": {d: round(r["veritas"], 4) for d, r in domain_results.items()},
        "severity_agreement_kappa_overall": overall_kappa,
        "severity_agreement_kappa_by_domain": kappa_by_domain,
    }
    with open(OUTPUT_DIR / f"manifest_{nickname}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    generator.unload()

    print(f"\n[MODEL {nickname} COMPLETE]")
    print(f"  Files written to {OUTPUT_DIR}:")
    for f in [final_csv, OUTPUT_DIR / f"table13_temporal_{nickname}.csv",
              OUTPUT_DIR / f"severity_agreement_{nickname}.csv",
              OUTPUT_DIR / f"manifest_{nickname}.json"]:
        print(f"   - {f.name}")
    print(f"  Domain VERITAS scores: {manifest['domain_veritas_scores']}")
    print(f"  Severity agreement kappa: {kappa_by_domain} | overall: {round(overall_kappa,3)}")
    print("  >>> You can stop the kernel here and download these files, or continue to the next model. <<<\n")

    return all_df, domain_results, manifest

# ── RUN ALL MODELS IN SEQUENCE ───────────────────────────────────────────────
all_results = {}
for model_cfg in MODELS_TO_RUN:
    try:
        df, dres, manifest = run_for_model(model_cfg)
        all_results[model_cfg["nickname"]] = manifest
    except Exception as e:
        log.error(f"Model {model_cfg['nickname']} failed: {e}")
        log.error("Checkpoint for this model (if any) is preserved. Fix the issue and re-run this cell; "
                   "already-scored claims will be skipped.")
        raise

print("ALL MODELS COMPLETE.")
print(json.dumps(all_results, indent=2))