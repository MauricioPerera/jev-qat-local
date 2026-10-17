"""Inspect actual constrained-label probabilities before implementing the adapter."""
import json
from pathlib import Path
import socket
import string
import subprocess
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "integration" / "probe"
OUT.mkdir(parents=True, exist_ok=True)
previous = json.loads((ROOT / "inference/qat-q4-vulkan/results.json").read_text(encoding="utf-8"))
arguments = previous["server_arguments"]
with socket.socket() as port_probe:
    port_probe.bind(("127.0.0.1", 0))
    port = port_probe.getsockname()[1]
arguments[arguments.index("--port") + 1] = str(port)
with (OUT / "server.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen(arguments, cwd=Path(arguments[0]).parent,
                               stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False) as client:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"Backend exited: {process.returncode}")
                try:
                    if client.get("/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Backend startup timeout")
            labels = []
            candidates = list(string.ascii_uppercase) + [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]
            for label in candidates:
                tokens = client.post("/tokenize", json={"content": label, "add_special": False}).json()["tokens"]
                if len(tokens) == 1 and tokens[0] not in {entry["id"] for entry in labels}:
                    labels.append({"label": label, "id": tokens[0]})
                if len(labels) == 256:
                    break
            (OUT / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
            print(f"Verified {len(labels)} unique single-token option labels", flush=True)
            payload = {
                "messages": [{"role": "user", "content": "STATE: The meeting starts at 10 AM. It is now 9 AM.\nQUESTION: Has the meeting started?\nOPTIONS:\nA. Yes\nB. No\nReply with only the correct option label."}],
                "max_tokens": 1, "temperature": 1, "top_k": 0, "top_p": 1,
                "min_p": 0, "repeat_penalty": 1, "samplers": ["temperature"],
                "logit_bias": {str(entry["id"]): 100 for entry in labels[:2]}, "logprobs": True,
                "top_logprobs": 2, "post_sampling_probs": True, "seed": 123,
            }
            response = client.post("/v1/chat/completions", json=payload)
            print("HTTP", response.status_code, flush=True)
            result = response.json()
            (OUT / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True)[:4000], flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
