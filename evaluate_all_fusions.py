import re
import time
import json
import math
import csv
import os
import requests
from typing import List, Dict, Tuple, Any

UNIFIED_URL = "http://localhost:8000/hackrx/run"
DOC = "https://mlopsbalti.s3.eu-north-1.amazonaws.com/xyz.pdf"

QUESTIONS = [
    "How many total orders were delivered?",
    "What is the total revenue generated?",
    "What is the average order value?",
    "What percentage of orders were cancelled?",
    "What is the main objective of the Mealawe project as stated in the introduction?",
    "List two customer behavior KPIs mentioned in the document.",
    "What operational gaps were identified in the problem statement?",
    "Which technology was chosen for the dashboard frontend and why?",
    "According to the literature review, which model helps determine if a customer has churned?",
    "What are the planned outcomes of this project?",
]

GROUND_TRUTH = {
    "How many total orders were delivered?": "10,297",
    "What is the total revenue generated?": "₹1,845,863.74",
    "What is the average order value?": "₹179.26",
    "What percentage of orders were cancelled?": "1.93%",
    "What is the main objective of the Mealawe project as stated in the introduction?": "To analyze and visualize customer behavioral and order fulfillment KPIs for Mealawe to improve decision-making.",
    "List two customer behavior KPIs mentioned in the document.": "Average Order Value (AOV), Order Frequency.",
    "What operational gaps were identified in the problem statement?": "High cancellation rates, delays in order deliveries, and lack of visibility into operational performance.",
    "Which technology was chosen for the dashboard frontend and why?": "ReactJS was chosen for its component reusability and ability to create interactive dashboards.",
    "According to the literature review, which model helps determine if a customer has churned?": "Survival analysis model.",
    "What are the planned outcomes of this project?": "Improved operational efficiency, better customer satisfaction, and real-time visibility of KPIs through an analytics dashboard.",
}


# ---------------- Text normalization helpers ----------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # normalize currency and punctuation spacing
    s = s.replace("₹", "rs ")
    s = re.sub(r"[^a-z0-9.% ]+", " ", s)  # keep alnum, percent, dot
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in s.split() if t]


def exact_match(a: str, b: str) -> int:
    return 1 if normalize_text(a) == normalize_text(b) else 0


def token_prf(pred: str, ref: str) -> Tuple[float, float, float]:
    pt = tokenize(pred)
    rt = tokenize(ref)
    if not pt and not rt:
        return 1.0, 1.0, 1.0
    if not pt or not rt:
        return 0.0, 0.0, 0.0
    pset = pt
    rset = rt
    # multiset overlap count
    from collections import Counter
    pc, rc = Counter(pset), Counter(rset)
    overlap = sum((pc & rc).values())
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(rc.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def bleu1(pred: str, ref: str) -> float:
    pt = tokenize(pred)
    rt = tokenize(ref)
    if not pt or not rt:
        return 0.0
    from collections import Counter
    pc, rc = Counter(pt), Counter(rt)
    overlap = sum((pc & rc).values())
    p1 = overlap / max(1, sum(pc.values()))
    bp = 1.0 if len(pt) > len(rt) else math.exp(1 - (len(rt) / max(1, len(pt))))
    return bp * p1


def lcs(a: List[str], b: List[str]) -> int:
    # classic LCS DP
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(pred: str, ref: str) -> float:
    pt = tokenize(pred)
    rt = tokenize(ref)
    if not pt or not rt:
        return 0.0
    L = lcs(pt, rt)
    prec = L / len(pt)
    rec = L / len(rt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------- Numeric-aware evaluation ----------------
num_pat = re.compile(r"([+-]?\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?%?")


def extract_first_number(s: str) -> Tuple[bool, float, bool]:
    """Return (found, value, is_percent) extracting the first numeric-like token.
    Handles thousand separators and optional percent sign.
    """
    if not s:
        return False, 0.0, False
    m = num_pat.search(s.replace("₹", ""))
    if not m:
        return False, 0.0, False
    token = m.group(0)
    is_percent = token.endswith("%")
    token = token.rstrip("%")
    token = token.replace(",", "")
    try:
        val = float(token)
        return True, val, is_percent
    except Exception:
        return False, 0.0, False


def numeric_metrics(pred: str, ref: str) -> Dict[str, Any]:
    rf, rv, r_pct = extract_first_number(ref)
    pf, pv, p_pct = extract_first_number(pred)
    out: Dict[str, Any] = {
        "numeric_applicable": bool(rf),
        "ref_value": rv if rf else None,
        "pred_value": pv if pf else None,
        "ref_is_percent": r_pct if rf else None,
        "pred_is_percent": p_pct if pf else None,
        "match": None,
        "abs_error": None,
        "rel_error": None,
    }
    if rf and pf:
        # if one is percent and other not, treat as not matching
        if r_pct != p_pct:
            out.update({"match": False})
            return out
        # tolerance: orders => 1; money/AOV => 0.01; percent => 0.01
        tol = 0.01
        if not r_pct and abs(rv) >= 1000:
            tol = 1.0
        abs_err = abs(pv - rv)
        rel_err = abs_err / rv if rv != 0 else (0.0 if abs_err == 0 else float("inf"))
        match = abs_err <= tol
        out.update({"match": match, "abs_error": abs_err, "rel_error": rel_err})
    elif rf and not pf:
        out.update({"match": False})
    return out


def call_mode(mode: str, questions: List[str]) -> Dict[str, Any]:
    payload = {"documents": DOC, "questions": questions, "fusion": mode}
    t0 = time.time()
    resp = requests.post(UNIFIED_URL, json=payload, timeout=240)
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    return {"latency": dt, "answers": data.get("answers", []), "debug": data.get("debug", {})}


def evaluate_mode(mode: str) -> Dict[str, Any]:
    out = call_mode(mode, QUESTIONS)
    answers = out["answers"]
    metrics = []
    # try to extract per-question debug if available
    per_q = out.get("debug", {}).get("per_q", []) if isinstance(out.get("debug", {}), dict) else []
    for q, pred in zip(QUESTIONS, answers):
        ref = GROUND_TRUTH[q]
        em = exact_match(pred, ref)
        p, r, f1 = token_prf(pred, ref)
        b1 = bleu1(pred, ref)
        rl = rouge_l(pred, ref)
        num = numeric_metrics(pred, ref)
        # optional retrieval metrics per q
        qdbg = per_q[QUESTIONS.index(q)] if QUESTIONS.index(q) < len(per_q) else {}
        metrics.append({
            "question": q,
            "pred": pred,
            "ref": ref,
            "exact_match": em,
            "precision": p,
            "recall": r,
            "f1": f1,
            "bleu1": b1,
            "rougeL": rl,
            "numeric": num,
            "retrieved": qdbg.get("k_retrieved") if isinstance(qdbg, dict) else None,
            "used_images": len(qdbg.get("used_images", [])) if isinstance(qdbg, dict) else None,
        })

    def avg(key: str) -> float:
        vals = [m[key] for m in metrics]
        return sum(vals) / max(1, len(vals))

    # Numeric subset aggregates
    num_items = [m["numeric"] for m in metrics if m.get("numeric", {}).get("numeric_applicable")]
    def navg(key: str) -> float:
        vals = [x[key] for x in num_items if x.get(key) is not None]
        return sum(vals) / max(1, len(vals)) if vals else 0.0
    nmatch = [1.0 if x.get("match") else 0.0 for x in num_items if x.get("match") is not None]
    nmatch_avg = sum(nmatch) / max(1, len(nmatch)) if nmatch else 0.0

    summary = {
        "latency": out["latency"],
        "exact_match": avg("exact_match"),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "bleu1": avg("bleu1"),
        "rougeL": avg("rougeL"),
        "numeric_match_rate": nmatch_avg,
        "numeric_abs_error": navg("abs_error"),
        "numeric_rel_error": navg("rel_error"),
    }

    return {"mode": mode, "summary": summary, "details": metrics, "raw": out}


def main():
    results = {}
    detailed_rows = []
    summary_rows = []
    for mode in ["late", "early", "hybrid"]:
        print(f"\n=== Evaluating {mode.upper()} ===")
        try:
            res = evaluate_mode(mode)
            results[mode] = res
            s = res["summary"]
            print(f"Latency: {s['latency']:.2f}s")
            print(f"EM: {s['exact_match']:.3f} | P: {s['precision']:.3f} | R: {s['recall']:.3f} | F1: {s['f1']:.3f} | BLEU1: {s['bleu1']:.3f} | ROUGE-L: {s['rougeL']:.3f}")

            # build detailed rows
            latency_ms = s["latency"] * 1000.0
            for idx, m in enumerate(res["details"]):
                # normalized fields
                pred_text_norm = normalize_text(m["pred"]) if isinstance(m["pred"], str) else ""
                gold_text_norm = normalize_text(m["ref"]) if isinstance(m["ref"], str) else ""
                pred_num_norm = m["numeric"].get("pred_value") if isinstance(m.get("numeric"), dict) else None
                gold_num_norm = m["numeric"].get("ref_value") if isinstance(m.get("numeric"), dict) else None
                detailed_rows.append({
                    "qid": f"Q{idx+1}",
                    "system": f"{mode.capitalize()} Fusion",
                    "question": m["question"],
                    "answer": m["pred"],
                    "confidence": results[mode]["raw"].get("debug", {}).get("confidence", None),
                    "latency_ms": latency_ms,
                    "pred_text_norm": pred_text_norm,
                    "gold_text_norm": gold_text_norm,
                    "pred_num_norm": pred_num_norm,
                    "gold_num_norm": gold_num_norm,
                    "EM": m["exact_match"],
                    "F1": m["f1"],
                    "ROUGE_L": m["rougeL"],
                    "METEOR": None,  # placeholder (not computed here)
                    "BLEU": m["bleu1"],
                    "Num_Acc@tol": 1.0 if m["numeric"].get("match") else 0.0 if m["numeric"].get("match") is not None else None,
                    "Num_MAE": m["numeric"].get("abs_error"),
                    "Num_RMSE": m["numeric"].get("abs_error"),
                    "Recall@5": results[mode]["raw"].get("debug", {}).get("recall@5", None),
                    "Recall@10": results[mode]["raw"].get("debug", {}).get("recall@10", None),
                    "MRR": results[mode]["raw"].get("debug", {}).get("mrr", None),
                    "error": results[mode]["raw"].get("debug", {}).get("error", None),
                })

            # summary row per system
            summary_rows.append({
                "system": f"{mode.capitalize()} Fusion",
                "confidence": results[mode]["raw"].get("debug", {}).get("confidence", None),
                "latency_ms": latency_ms,
                "pred_num_norm": sum([row["pred_num_norm"] or 0 for row in detailed_rows[-len(QUESTIONS):]])/len(QUESTIONS),
                "gold_num_norm": sum([row["gold_num_norm"] or 0 for row in detailed_rows[-len(QUESTIONS):]])/len(QUESTIONS),
                "EM": s["exact_match"],
                "F1": s["f1"],
                "ROUGE_L": s["rougeL"],
                "METEOR": None,
                "BLEU": s["bleu1"],
                "Num_Acc@tol": s["numeric_match_rate"],
                "Num_MAE": s["numeric_abs_error"],
                "Num_RMSE": s["numeric_abs_error"],
                "Recall@5": results[mode]["raw"].get("debug", {}).get("recall@5", None),
                "Recall@10": results[mode]["raw"].get("debug", {}).get("recall@10", None),
                "MRR": results[mode]["raw"].get("debug", {}).get("mrr", None),
            })
        except Exception as e:
            print(f"Error in {mode}: {e}")

    with open("fusion_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved fusion_benchmark_results.json with detailed metrics.")

    # save CSVs
    detailed_cols = [
        "qid","system","question","answer","confidence","latency_ms","pred_text_norm","gold_text_norm",
        "pred_num_norm","gold_num_norm","EM","F1","ROUGE_L","METEOR","BLEU","Num_Acc@tol","Num_MAE","Num_RMSE",
        "Recall@5","Recall@10","MRR","error"
    ]
    summary_cols = [
        "system","confidence","latency_ms","pred_num_norm","gold_num_norm","EM","F1","ROUGE_L","METEOR","BLEU",
        "Num_Acc@tol","Num_MAE","Num_RMSE","Recall@5","Recall@10","MRR"
    ]
    with open("fusion_eval_detailed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detailed_cols)
        w.writeheader()
        for row in detailed_rows:
            w.writerow(row)
    with open("fusion_eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    # Save configuration for reproducibility
    config = {
        "document": DOC,
        "unified_url": UNIFIED_URL,
        "questions": QUESTIONS,
        "timestamp": time.time(),
        "env": {k: os.getenv(k) for k in [
            "MODEL_DEVICE","MODEL_DEVICE_ID","TEXT_EMB_MODEL_NAME","CLIP_MODEL_NAME","CROSS_ENCODER_MODEL",
            "FAISS_USE_GPU","GEMINI_MODEL","GEMINI_INCLUDE_IMAGES"
        ]},
    }
    with open("fusion_eval_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Saved fusion_eval_detailed.csv, fusion_eval_summary.csv, and fusion_eval_config.json")


if __name__ == "__main__":
    main()
