#!/usr/bin/env python3
"""Smoke test for a running llama-server: 18 checks over HTTP, standard library only.

Start the server, then point this at it:

    ./start_llama.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --port 8089 &
    python utils/test_server_api.py                       # http://127.0.0.1:8089
    python utils/test_server_api.py --url http://host:8080

Checks: /health, /v1/models, /props; /completion (answer, n_predict, determinism at temperature 0);
/v1/completions; /v1/chat/completions (system message, token usage, multi-turn memory, streaming SSE);
/tokenize and /detokenize; four concurrent requests (continuous batching); error handling (bad JSON,
unknown route, malformed chat request) and that the server is still healthy afterwards.

The answer checks (Paris, Rome, Berlin, Madrid, Tokyo, "Alex") assume a chat-capable model such as
BitNet-b1.58-2B-4T, whose chat template start_llama.py / run_inference_server.py select automatically; a
much smaller or base-only model can fail them on content. Exit status 0 means every check passed.
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

_ap = argparse.ArgumentParser(description="Smoke test for a running llama-server.")
_ap.add_argument("--url", default="http://127.0.0.1:8089", help="server base URL (default: %(default)s)")
BASE = _ap.parse_args().url.rstrip("/")
results = []


def req(method, path, body=None, raw_body=None, timeout=180):
    data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            b = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        b, code = e.read(), e.code
    try:
        return code, json.loads(b)
    except ValueError:
        return code, b


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  - " + detail) if detail else ""))
    results.append(bool(ok))


try:
    urllib.request.urlopen(BASE + "/health", timeout=10).close()
except urllib.error.HTTPError:
    pass  # reachable: the checks below report what the server said
except (urllib.error.URLError, OSError) as e:
    print("cannot reach the server at %s (%s); start it first, e.g. ./start_llama.py -m MODEL.gguf --port 8089" % (BASE, getattr(e, "reason", e)))
    sys.exit(2)

# 1. health
code, j = req("GET", "/health")
check("GET /health", code == 200 and j.get("status") == "ok", "%s %s" % (code, j))

# 2. models / props
code, j = req("GET", "/v1/models")
ids = [m.get("id") for m in j.get("data", [])] if isinstance(j, dict) else []
check("GET /v1/models lists a model", code == 200 and len(ids) >= 1, str(ids)[:80])
code, j = req("GET", "/props")
check("GET /props", code == 200 and isinstance(j, dict) and ("default_generation_settings" in j or "model_path" in j), "keys: " + ",".join(list(j)[:5]) if isinstance(j, dict) else str(j)[:60])

# 3. native completion, deterministic
t = time.time()
code, j = req("POST", "/completion", {"prompt": "The capital of France is", "n_predict": 12, "temperature": 0})
dt = time.time() - t
text = j.get("content", "") if isinstance(j, dict) else ""
tm = j.get("timings", {}) if isinstance(j, dict) else {}
check("POST /completion answers 'Paris'", code == 200 and "Paris" in text, "%r  [%.1f t/s generation, %.1f s total]" % (text[:50], tm.get("predicted_per_second", 0), dt))
check("POST /completion respects n_predict", isinstance(j, dict) and j.get("tokens_predicted", 99) <= 12, "tokens_predicted=%s" % (j.get("tokens_predicted") if isinstance(j, dict) else "?"))

# 4. determinism at temperature 0
code2, j2 = req("POST", "/completion", {"prompt": "The capital of France is", "n_predict": 12, "temperature": 0})
check("temperature 0 is deterministic", code2 == 200 and j2.get("content") == text)

# 5. OpenAI-style completions
code, j = req("POST", "/v1/completions", {"prompt": "The capital of Italy is", "max_tokens": 10, "temperature": 0})
txt = j["choices"][0]["text"] if code == 200 and isinstance(j, dict) and j.get("choices") else ""
check("POST /v1/completions answers 'Rome'", code == 200 and "Rome" in txt, repr(txt[:50]))

# 6. chat completions (non-streaming), with a system message
body = {"messages": [{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "What is the capital of France? Answer in one short sentence."}], "max_tokens": 40, "temperature": 0}
code, j = req("POST", "/v1/chat/completions", body)
msg = j["choices"][0]["message"]["content"] if code == 200 and isinstance(j, dict) and j.get("choices") else ""
fin = j["choices"][0].get("finish_reason") if code == 200 and isinstance(j, dict) and j.get("choices") else None
check("POST /v1/chat/completions answers 'Paris'", code == 200 and "Paris" in msg, "%r finish=%s" % (msg[:70], fin))
usage = j.get("usage", {}) if isinstance(j, dict) else {}
check("chat response reports token usage", usage.get("completion_tokens", 0) > 0 and usage.get("prompt_tokens", 0) > 0, str(usage))

# 7. multi-turn chat remembers context
body = {"messages": [{"role": "user", "content": "My name is Alex."}, {"role": "assistant", "content": "Nice to meet you, Alex."}, {"role": "user", "content": "What is my name?"}], "max_tokens": 20, "temperature": 0}
code, j = req("POST", "/v1/chat/completions", body)
msg = j["choices"][0]["message"]["content"] if code == 200 and isinstance(j, dict) and j.get("choices") else ""
check("multi-turn chat remembers the name", code == 200 and "Alex" in msg, repr(msg[:60]))

# 8. streaming chat (server-sent events)
body = {"messages": [{"role": "user", "content": "Count from one to five."}], "max_tokens": 30, "temperature": 0, "stream": True}
r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
chunks, parts, done, first = 0, [], False, None
t0 = time.time()
with urllib.request.urlopen(r, timeout=180) as resp:
    ctype = resp.headers.get("Content-Type", "")
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        d = json.loads(payload)
        delta = d["choices"][0].get("delta", {}).get("content")
        if delta:
            if first is None:
                first = time.time() - t0
            chunks += 1
            parts.append(delta)
check("streaming chat returns SSE chunks and [DONE]", "event-stream" in ctype and chunks >= 3 and done, "%d chunks, first token after %.2f s, text %r" % (chunks, first or -1, "".join(parts)[:50]))

# 9. tokenize / detokenize round trip
code, j = req("POST", "/tokenize", {"content": "Hello world"})
toks = j.get("tokens", []) if isinstance(j, dict) else []
check("POST /tokenize", code == 200 and len(toks) >= 2 and all(isinstance(x, int) for x in toks), str(toks))
code, j = req("POST", "/detokenize", {"tokens": toks})
check("POST /detokenize round-trips", code == 200 and "Hello world" in j.get("content", ""), repr(j.get("content") if isinstance(j, dict) else j))

# 10. concurrent requests (continuous batching)
prompts = ["The capital of Italy is", "The capital of Germany is", "The capital of Spain is", "The capital of Japan is"]
expect = ["Rome", "Berlin", "Madrid", "Tokyo"]
def one(p):
    return req("POST", "/completion", {"prompt": p, "n_predict": 8, "temperature": 0})
t = time.time()
with cf.ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(one, prompts))
dt = time.time() - t
ok_codes = all(c == 200 for c, _ in outs)
right = sum(1 for (c, j), e in zip(outs, expect) if c == 200 and e in j.get("content", ""))
check("4 concurrent /completion requests all succeed", ok_codes, "%d/4 answers correct, %.1f s wall" % (right, dt))

# 11. error handling
code, j = req("POST", "/completion", raw_body=b"{not json")
check("invalid JSON is rejected (4xx/5xx with an error body)", code >= 400 and code < 600, "%s %s" % (code, str(j)[:70]))
code, j = req("GET", "/no-such-route")
check("unknown route returns 404", code == 404, str(code))
code, j = req("POST", "/v1/chat/completions", {"messages": "not a list"})
check("malformed chat request is rejected", code >= 400, "%s" % code)

# 12. server still healthy after the bad requests
code, j = req("GET", "/health")
check("server healthy after error cases", code == 200 and j.get("status") == "ok")

print("\n%d/%d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
