# eval/run_eval.py
import argparse
import json
import math
import re
import time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import urllib.request, urllib.error

# Additional imports for ROUGE-L, METEOR, and BLEU
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

# ---------------- Config ----------------
SYSTEMS = {
    "early":  {"name": "Early Fusion",  "endpoint": "http://localhost:8001/ask"},
    "late":   {"name": "Late Fusion",   "endpoint": "http://localhost:8002/ask"},
    "hybrid": {"name": "Hybrid Fusion", "endpoint": "http://localhost:8003/ask"},
}

TIMEOUT_SEC = 120
REL_TOL = 0.05   # numeric relative tolerance (5%)  <-- relaxed
ABS_TOL = 1e-6   # absolute tolerance

# ---------------- Normalization helpers ----------------
_WS = re.compile(r"\s+")
_CURRENCY = re.compile(r"(₹|rs\.?|inr|\$|usd)", re.I)
_SUFFIX = re.compile(r"(k|m|mn|b|bn)\b", re.I)
_NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\.\d+|[-+]?\d+")

def normalize_text_simple(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation space-only edges."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = _WS.sub(" ", s)
    return s

def to_float_with_suffix(s: str) -> Optional[float]:
    """
    Extract a numeric value from a string that may include currency, commas,
    percent signs, and K/M/B suffixes. Returns float or None.
    """
    if not s:
        return None
    raw = s.lower().strip()
    # Remove currency words/symbols
    raw = _CURRENCY.sub("", raw)
    # Remove percent sign (we'll treat % in caller via expect_pct flag)
    raw = raw.replace("%", "")
    # Remove commas
    raw = raw.replace(",", "")

    mult = 1.0
    m = _SUFFIX.search(raw)
    if m:
        suf = m.group(1).lower()
        if suf == "k":
            mult = 1e3
        elif suf in ("m", "mn"):
            mult = 1e6
        elif suf in ("b", "bn"):
            mult = 1e9
        raw = _SUFFIX.sub("", raw)

    m2 = re.search(r"[-+]?\d*\.?\d+", raw)
    if not m2:
        return None
    try:
        return float(m2.group(0)) * mult
    except Exception:
        return None

def extract_first_numeric(s: str, expect_pct: bool=False) -> Optional[float]:
    """
    Extract the first numeric value; if expect_pct, return the number as-is
    (assuming caller will compare to 0-100 scale).
    """
    if not s:
        return None
    # Use suffix-aware conversion on the first match
    m = _NUM.search(s)
    if not m:
        return to_float_with_suffix(s)
    # Apply normalization on the matched span as well (handles commas/suffix)
    return to_float_with_suffix(m.group(0))

def is_percent_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["percent", "%", "percentage", "rate", "ratio"])

# ---------------- API Caller ----------------
def call_system(endpoint: str, question: str, doc_path: str) -> Dict[str, Any]:
    payload = json.dumps({"question": question, "doc_path": doc_path}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}", "latency_ms": (time.time()-start)*1000}
    except Exception as e:
        return {"error": f"RequestError: {e}", "latency_ms": (time.time()-start)*1000}

    data["latency_ms"] = data.get("latency_ms", (time.time()-start)*1000)
    return data

# ---------------- Metrics ----------------
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text_simple(pred) == normalize_text_simple(gold) else 0.0

def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_text_simple(pred), normalize_text_simple(gold)
    p_toks, g_toks = p.split(), g.split()
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    num_same = sum(min(p_toks.count(t), g_toks.count(t)) for t in common)
    prec, rec = num_same/len(p_toks), num_same/len(g_toks)
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def numeric_accuracy(pred: Optional[float], gold: Optional[float]) -> float:
    if pred is None or gold is None:
        return 0.0
    try:
        return 1.0 if math.isclose(pred, gold, rel_tol=REL_TOL, abs_tol=ABS_TOL) else 0.0
    except Exception:
        return 0.0

def numeric_mae(pred, gold):
    return abs(pred-gold) if pred is not None and gold is not None else np.nan

def numeric_rmse(pred, gold):
    return math.sqrt((pred-gold)**2) if pred is not None and gold is not None else np.nan

def recall_at_k(cands: List[Dict[str, Any]], gold_page: Optional[int], k: int) -> float:
    if not gold_page or not cands:
        return np.nan
    pages = [c.get("page") for c in cands[:k] if "page" in c]
    return 1.0 if gold_page in pages else 0.0

def reciprocal_rank(cands: List[Dict[str, Any]], gold_page: Optional[int]) -> float:
    if not gold_page or not cands:
        return np.nan
    for i, c in enumerate(cands, 1):
        if c.get("page") == gold_page:
            return 1/i
    return 0.0

def rouge_l_score(pred: str, gold: str) -> float:
    """Calculate ROUGE-L score between predicted and gold text."""
    if not ROUGE_AVAILABLE:
        return np.nan
    
    pred_norm = normalize_text_simple(pred)
    gold_norm = normalize_text_simple(gold)
    
    if not pred_norm or not gold_norm:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(gold_norm, pred_norm)
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
        return np.nan

def meteor_score_func(pred: str, gold: str) -> float:
    """Calculate METEOR score between predicted and gold text."""
    if not NLTK_AVAILABLE:
        return np.nan
    
    pred_norm = normalize_text_simple(pred)
    gold_norm = normalize_text_simple(gold)
    
    if not pred_norm or not gold_norm:
        return 0.0
    
    try:
        # METEOR expects tokenized inputs
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()
        
        if not pred_tokens or not gold_tokens:
            return 0.0
            
        score = meteor_score([gold_tokens], pred_tokens)
        return score
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
        return np.nan

def bleu_score_func(pred: str, gold: str) -> float:
    """Calculate BLEU score between predicted and gold text."""
    if not NLTK_AVAILABLE:
        return np.nan
    
    pred_norm = normalize_text_simple(pred)
    gold_norm = normalize_text_simple(gold)
    
    if not pred_norm or not gold_norm:
        return 0.0
    
    try:
        # BLEU expects tokenized inputs
        pred_tokens = pred_norm.split()
        gold_tokens = [gold_norm.split()]  # Reference should be a list of lists
        
        if not pred_tokens or not gold_tokens[0]:
            return 0.0
        
        # Use smoothing to handle cases with no matching n-grams
        smoothing = SmoothingFunction()
        score = sentence_bleu(gold_tokens, pred_tokens, 
                            smoothing_function=smoothing.method1)
        return score
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return np.nan

# ---------------- Runner ----------------
def evaluate(queries_csv: str, out_csv: str, summary_csv: str):
    df = pd.read_csv(queries_csv)
    rows = []

    for _, r in df.iterrows():
        qid = r["qid"]
        doc = r["doc_path"]
        q = r["question"]
        ans_type = str(r.get("answer_type", "text")).strip().lower()

        # Golds (normalized)
        gold_text = str(r.get("answer_text", "") or "")
        gold_num = None
        if str(r.get("answer_numeric", "")).strip():
            try:
                gold_num = float(str(r["answer_numeric"]).replace(",", ""))
            except Exception:
                gold_num = None
        gold_page = int(r["answer_page"]) if str(r.get("answer_page", "")).strip().isdigit() else None

        for key, sysinfo in SYSTEMS.items():
            resp = call_system(sysinfo["endpoint"], q, doc)
            err = resp.get("error")
            answer_raw = resp.get("answer", "") if not err else ""
            latency = float(resp.get("latency_ms", 0))
            conf = resp.get("confidence")

            # Candidates (sorted by score desc if present)
            cands = resp.get("candidates")
            if isinstance(cands, list) and cands and "score" in cands[0]:
                cands = sorted(cands, key=lambda x: x.get("score", 0), reverse=True)

            # Normalized views (for debugging in CSV)
            pred_text_norm = normalize_text_simple(answer_raw)
            gold_text_norm = normalize_text_simple(gold_text)

            pred_num_norm = None
            gold_num_norm = gold_num

            # If this is a numeric question OR the question phrasing suggests numeric/percent
            if ans_type == "numeric" or is_percent_question(q):
                pred_num_norm = extract_first_numeric(answer_raw, expect_pct=is_percent_question(q))
                # Special case: if gold_num is percentage (0-100) and pred has 0-1 scale, or vice versa
                # We won't rescale automatically; assume both are 0..100 as per guidance

            # Metrics
            em = f1 = np.nan
            rouge_l = meteor = bleu = np.nan
            n_acc = n_mae = n_rmse = np.nan

            if ans_type == "text":
                em, f1 = exact_match(answer_raw, gold_text), f1_score(answer_raw, gold_text)
                # Calculate additional text metrics
                rouge_l = rouge_l_score(answer_raw, gold_text)
                meteor = meteor_score_func(answer_raw, gold_text)
                bleu = bleu_score_func(answer_raw, gold_text)
            else:
                # Numeric comparison
                n_acc = numeric_accuracy(pred_num_norm, gold_num_norm)
                n_mae = numeric_mae(pred_num_norm, gold_num_norm)
                n_rmse = numeric_rmse(pred_num_norm, gold_num_norm)

            r5 = r10 = mrr = np.nan
            if isinstance(cands, list):
                r5, r10 = recall_at_k(cands, gold_page, 5), recall_at_k(cands, gold_page, 10)
                mrr = reciprocal_rank(cands, gold_page)

            rows.append({
                "qid": qid,
                "system": sysinfo["name"],
                "question": q,
                "answer": answer_raw,
                "confidence": conf,
                "latency_ms": latency,
                # Debug normals
                "pred_text_norm": pred_text_norm,
                "gold_text_norm": gold_text_norm,
                "pred_num_norm": pred_num_norm,
                "gold_num_norm": gold_num_norm,
                # Metrics
                "EM": em,
                "F1": f1,
                "ROUGE_L": rouge_l,
                "METEOR": meteor,
                "BLEU": bleu,
                "Num_Acc@tol": n_acc,
                "Num_MAE": n_mae,
                "Num_RMSE": n_rmse,
                "Recall@5": r5,
                "Recall@10": r10,
                "MRR": mrr,
                "error": err
            })

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)

    # Summary by system (ignore non-numeric cols)
    summary = res.groupby("system").mean(numeric_only=True).reset_index()
    summary.to_csv(summary_csv, index=False)

    print("Saved detailed results to", out_csv)
    print("Saved summary to", summary_csv)
    print("\nPer-system summary:")
    print(summary[["system","EM","F1","ROUGE_L","METEOR","BLEU","Num_Acc@tol","Num_MAE","Num_RMSE","Recall@5","MRR","confidence","latency_ms"]])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="eval/queries_small.csv")
    p.add_argument("--out", default="eval/results_small.csv")
    p.add_argument("--summary", default="eval/summary_small.csv")
    args = p.parse_args()
    evaluate(args.queries, args.out, args.summary)
