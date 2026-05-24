# eval/quick_validate.py
import json, urllib.request, sys

FUSIONS = {
  "early":  "http://localhost:8001",
  "late":   "http://localhost:8002",
  "hybrid": "http://localhost:8003",
}

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def post(url, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    ok = True
    for name, base in FUSIONS.items():
        try:
            h = get(f"{base}/health")
            assert h.get("status") == "ok"
            print(f"[✓] {name} /health ok")
        except Exception as e:
            ok = False
            print(f"[x] {name} /health failed:", e)

    probe = {
        "documents": ["https://mlopsbalti.s3.eu-north-1.amazonaws.com/xyz.pdf"],
        "question": "Which technology was chosen for the dashboard frontend and why?",
        "return_candidates": True
    }
    for name, base in FUSIONS.items():
        try:
            r = post(f"{base}/ask", probe)
            for k in ["answer","confidence","evidence","candidates","latency_ms"]:
                assert k in r, f"missing key {k}"
            print(f"[✓] {name} /ask ok — conf={round(r['confidence'],3)}  evidence_n={len(r['evidence'])}")
        except Exception as e:
            ok = False
            print(f"[x] {name} /ask failed:", e)

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
