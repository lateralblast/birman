#!/usr/bin/env python3
"""Minimal retrieval-augmented generation against two running llama-servers (Python standard library only).

  ./start_llama.py --open                 # chat model, port 8080 (any chat model)
  ./start_llama.py --embedding --open     # embedding model (BitNet-embedding-0.6B), port 8081

  python utils/rag_demo.py --docs README.md docs/ --host SERVER --api-key-file KEYFILE "How is the API key stored?"

Documents (files, or directories searched for *.md/*.txt/*.rst) are split into chunks of about --chunk-chars characters
on paragraph boundaries and embedded in batches; the question is embedded with the model's query instruction
(documents get none, as the BitNet-embedding model card says); the --top-k most similar chunks (cosine similarity, the
server returns normalised vectors) go into the prompt of the chat model, which is told to answer only from them.
Everything is held in memory: the index is rebuilt on every run, which is fine for a few thousand chunks.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

TASK = "Given a question, retrieve passages that answer the question"


def post(url, body, key, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit("error: %s -> HTTP %d %s%s" % (url, e.code, e.read()[:200].decode("utf-8", "replace"),
                                                 " (is the API key right? --api-key / --api-key-file)" if e.code == 401 else ""))
    except (urllib.error.URLError, OSError) as e:
        sys.exit("error: cannot reach %s (%s)" % (url, getattr(e, "reason", e)))


CACHE = os.path.join(os.path.expanduser("~"), ".cache", "start_llama", "rag_vectors.json")


def load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def files_under(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _d, names in os.walk(p):
                for n in sorted(names):
                    if n.endswith((".md", ".txt", ".rst")):
                        yield os.path.join(root, n)
        else:
            yield p


def chunks_of(text, limit):
    """Split on blank lines, merge small paragraphs up to limit characters, cut oversized ones."""
    cur = ""
    for para in (x.strip() for x in text.split("\n\n")):
        if not para:
            continue
        while len(para) > limit:
            if cur:
                yield cur
                cur = ""
            yield para[:limit]
            para = para[limit:]
        if cur and len(cur) + len(para) + 2 > limit:
            yield cur
            cur = ""
        cur = cur + "\n\n" + para if cur else para
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser(description="Minimal RAG demo: embedding server + chat server.")
    ap.add_argument("question")
    ap.add_argument("--docs", nargs="+", required=True, help="files or directories (*.md, *.txt, *.rst)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--embed-port", type=int, default=8081)
    ap.add_argument("--chat-port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    ap.add_argument("--api-key-file")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--chunk-chars", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true", help="re-embed every chunk (vectors are otherwise cached by chunk text and model in %s)" % CACHE)
    ap.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    a = ap.parse_args()
    key = a.api_key
    if a.api_key_file:
        with open(a.api_key_file) as f:
            key = next(ln.strip() for ln in f if ln.strip() and not ln.startswith("#"))
    embed_url = "http://%s:%d/v1/embeddings" % (a.host, a.embed_port)
    chat_url = "http://%s:%d/v1/chat/completions" % (a.host, a.chat_port)

    chunks = []  # (source, text)
    for path in files_under(a.docs):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                chunks += [(path, c) for c in chunks_of(f.read(), a.chunk_chars)]
        except OSError as e:
            print("skipping %s: %s" % (path, e), file=sys.stderr)
    if not chunks:
        sys.exit("error: no text found under %s" % a.docs)

    t = time.time()
    cache = {} if a.no_cache else load_cache()
    model = post(embed_url, {"input": ["x"]}, key).get("model", "")  # the cache key includes the model that made a vector
    keys = [hashlib.sha1((model + "\0" + c).encode()).hexdigest() for _s, c in chunks]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    for j in range(0, len(todo), 16):
        batch = todo[j:j + 16]
        out = post(embed_url, {"input": [chunks[i][1] for i in batch]}, key)
        for i, d in zip(batch, sorted(out["data"], key=lambda d: d["index"])):
            cache[keys[i]] = d["embedding"]
    if todo and not a.no_cache:
        save_cache(cache)
    vecs = [cache[k] for k in keys]
    print("indexed %d chunks from %d files in %.1f s (%d embedded, %d from cache)"
          % (len(chunks), len({s for s, _ in chunks}), time.time() - t, len(todo), len(chunks) - len(todo)), file=sys.stderr)

    q = post(embed_url, {"input": ["Instruct: %s\nQuery: %s" % (TASK, a.question)]}, key)["data"][0]["embedding"]
    scores = sorted(((sum(x * y for x, y in zip(q, v)), i) for i, v in enumerate(vecs)), reverse=True)[:a.top_k]
    context = "\n\n".join("[%d] (%s)\n%s" % (n + 1, chunks[i][0], chunks[i][1]) for n, (_s, i) in enumerate(scores))
    if a.show_context:
        for n, (s, i) in enumerate(scores):
            print("[%d] %.3f %s: %s" % (n + 1, s, chunks[i][0], chunks[i][1][:120].replace("\n", " ")), file=sys.stderr)

    prompt = ("Answer the question using only the context below. If the context does not contain the answer, say so. "
              "Cite the passage numbers you used.\n\nContext:\n%s\n\nQuestion: %s" % (context, a.question))
    r = post(chat_url, {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens, "temperature": 0}, key)
    print(r["choices"][0]["message"]["content"].strip())
    print("\nSources: " + ", ".join("[%d] %s" % (n + 1, chunks[i][0]) for n, (_s, i) in enumerate(scores)), file=sys.stderr)


if __name__ == "__main__":
    main()
