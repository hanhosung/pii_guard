"""
Real end-to-end smoke test for PII-Guard.

Launches a mock upstream (captures what the proxy forwards) + the real
`piiguard serve` proxy subprocess, then sends Claude Messages requests through
it to verify, against live code (incl. the lg NER engine):

  1. MASK  — Korean PII (name/phone) is replaced with placeholders before the
             request reaches the upstream.
  2. BLOCK — a hard secret (AWS key) is blocked; the upstream is never called.
  3. REHYDRATE — placeholders in the upstream response are restored for the client.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pathlib
REPO = str(pathlib.Path(__file__).resolve().parent.parent)

# ── Mock upstream — captures the body the proxy forwards ──────────────────────
captured = {"body": None, "calls": 0}


class MockUpstream(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        captured["body"] = raw.decode("utf-8", "replace")
        captured["calls"] += 1
        # Echo a Claude-style response that contains a placeholder, so we can
        # verify the proxy rehydrates it back to the real value for the client.
        try:
            sent = json.loads(raw)
            # find a PERSON placeholder in what we received, echo it back
        except Exception:
            sent = {}
        resp = {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello [PERSON_1], your request was received."}],
            "model": "mock",
            "stop_reason": "end_turn",
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_mock():
    srv = HTTPServer(("127.0.0.1", 0), MockUpstream)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main():
    _, mock_port = start_mock()
    print(f"[setup] mock upstream on :{mock_port}")

    proc = subprocess.Popen(
        [f"{REPO}/.venv/bin/python", "-m", "pii_guard.cli", "serve",
         "--upstream-url", f"http://127.0.0.1:{mock_port}", "--port", "0"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONPATH": REPO},
    )
    # wait for "READY <port>"
    proxy_port = None
    t0 = time.time()
    while time.time() - t0 < 60:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                print("[fail] proxy exited early:", proc.returncode)
                return 1
            continue
        line = line.strip()
        if line.startswith("READY"):
            proxy_port = int(line.split()[1])
            break
    if not proxy_port:
        print("[fail] proxy never became READY")
        proc.terminate(); return 1
    base = f"http://127.0.0.1:{proxy_port}"
    print(f"[setup] proxy READY on :{proxy_port}\n")

    results = []

    # ── Test 1: MASK Korean PII ───────────────────────────────────────────────
    captured["body"] = None; captured["calls"] = 0
    pii_text = "안녕하세요, 제 이름은 김민수이고 전화번호는 010-1234-5678 입니다."
    status, body = post(base + "/v1/messages", {
        "model": "claude-3-5-sonnet", "max_tokens": 100,
        "messages": [{"role": "user", "content": pii_text}],
    })
    fwd = captured["body"] or ""
    masked_ok = (captured["calls"] == 1 and "김민수" not in fwd
                 and "010-1234-5678" not in fwd
                 and ("[PERSON_1]" in fwd or "PERSON" in fwd))
    results.append(("MASK Korean PII (name/phone → placeholder before upstream)", masked_ok))
    print(f"[test1] client status={status}, upstream calls={captured['calls']}")
    print(f"        forwarded-to-upstream contains '김민수'? {'김민수' in fwd}  "
          f"contains placeholder? {'PERSON' in fwd}")
    # rehydration: client response should restore [PERSON_1] → 김민수
    rehydrated_ok = "김민수" in body and "[PERSON_1]" not in body
    results.append(("REHYDRATE (upstream [PERSON_1] → 김민수 in client response)", rehydrated_ok))
    print(f"        client response restores 김민수? {'김민수' in body}\n")

    # ── Test 2: BLOCK hard secret ─────────────────────────────────────────────
    captured["body"] = None; captured["calls"] = 0
    secret_text = "Use AWS key AKIAIOSFODNN7EXAMPLE to deploy."
    status, body = post(base + "/v1/messages", {
        "model": "claude-3-5-sonnet", "max_tokens": 100,
        "messages": [{"role": "user", "content": secret_text}],
    })
    blocked_ok = (captured["calls"] == 0 and status >= 400)
    results.append(("BLOCK AWS secret (4xx, upstream never called)", blocked_ok))
    print(f"[test2] client status={status}, upstream calls={captured['calls']} "
          f"(expect 0)\n")

    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 64)
    allok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok = allok and ok
    print("=" * 64)
    print("RESULT:", "ALL PASS ✅" if allok else "SOME FAILED ❌")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
