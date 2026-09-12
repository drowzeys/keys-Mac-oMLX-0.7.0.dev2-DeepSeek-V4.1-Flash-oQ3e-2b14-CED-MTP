"""Speed + concurrency bench for DeepSeek-V4.1-Flash with SSD-paged routed experts.

Notes on correctness of the measurement:
* V4.1 is a reasoning model: generated text arrives as delta.reasoning_content
  before any delta.content, so both must be counted.
* The server interleaves model="keepalive" chunks with empty content; counting
  those as the first token destroys TTFT.
* A chunk may carry several tokens, so chunk-counting undercounts. Exact counts
  come from usage.completion_tokens via stream_options.include_usage.
"""
import json, sys, time, urllib.request, threading, statistics as st

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11600"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "dsv41-flash-oq3e"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PROMPTS = {
    "prose": "Write a vivid 400-word short story about a lighthouse keeper who discovers "
             "the sea has started running backwards. Literary prose, no lists.",
    "code":  "Write a complete Python module implementing an LRU cache with TTL expiry, "
             "thread safety, and a decorator API. Include docstrings.",
}

def stream(prompt):
    body = json.dumps({"model": MODEL, "stream": True, "max_tokens": MAXTOK,
                       "temperature": 0.0,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            if d.get("model") == "keepalive":      # heartbeat, not generation
                continue
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    total = time.perf_counter() - t0
    n = (usage or {}).get("completion_tokens") or chunks
    decode = (n - 1) / (total - ttft) if ttft and n > 1 and total > ttft else 0.0
    return {"ttft": ttft or 0.0, "tokens": n, "total": total, "decode": decode}

def run(conc, prompt):
    out, lock = [], threading.Lock()
    def worker():
        r = stream(prompt)
        with lock: out.append(r)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    tok = sum(r["tokens"] for r in out)
    return {"concurrency": conc, "wall": wall, "tokens": tok,
            "aggregate": tok / wall if wall else 0.0,
            "per_stream": st.median([r["decode"] for r in out]),
            "ttft": st.median([r["ttft"] for r in out])}

print("warmup...", flush=True)
stream(PROMPTS["prose"])
rows = []
print(f"{'task':6s}{'conc':>5s}{'agg tok/s':>11s}{'per-stream':>12s}{'ttft s':>9s}{'tokens':>8s}{'wall s':>8s}", flush=True)
for tag, prompt in PROMPTS.items():
    for conc in (1, 2, 4, 8):
        r = run(conc, prompt); r["task"] = tag; rows.append(r)
        print(f"{tag:6s}{conc:>5d}{r['aggregate']:>11.2f}{r['per_stream']:>12.2f}"
              f"{r['ttft']:>9.2f}{r['tokens']:>8d}{r['wall']:>8.1f}", flush=True)
json.dump(rows, open("/Users/cityhunter/dsv41-work/logs/bench.json", "w"), indent=1)
print("bench complete", flush=True)
