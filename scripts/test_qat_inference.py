"""Run local text, image and audio smoke tests with the downloaded Gemma QAT."""

import base64
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import unicodedata
import wave

import httpx
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "qat-q4-vulkan"
MODEL_DIR = ROOT / "models" / "gemma-4-E4B-it-QAT-Q4_0"
SERVER = ROOT / "runtimes" / "llama-b11669-vulkan" / "llama-server.exe"


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def image_fixture():
    path = OUT / "test-image.png"
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 54)
    draw.text((235, 25), "5831", font=font, fill="black")
    draw.ellipse((45, 155, 165, 275), fill=(0, 70, 230))
    draw.polygon([(320, 145), (250, 275), (390, 275)], fill=(255, 215, 0))
    draw.rectangle((465, 155, 585, 275), fill=(230, 25, 35))
    image.save(path)
    return path


def assess(result):
    if result["http_status"] != 200 or result.get("finish_reason") != "stop":
        return False
    output = result.get("output") or ""
    normalized = "".join(char for char in unicodedata.normalize("NFKD", output.lower())
                         if not unicodedata.combining(char))
    if result["id"] == "text":
        return re.findall(r"\d+", output) == [str(index) for index in range(1, 21)]
    if result["id"] == "image":
        figures = [re.search(pattern, normalized, re.S) for pattern in
                   (r"circulo.{0,60}azul", r"triangulo.{0,60}amarillo", r"cuadrado.{0,60}rojo")]
        return ("5831" in normalized and all(figures)
                and [match.start() for match in figures] == sorted(match.start() for match in figures))
    if result["id"] == "audio":
        normalized = normalized.replace("cuarenta y dos", "42")
        return " ".join(re.findall(r"\w+", normalized)) == "el codigo de prueba es 42"
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = image_fixture()
    audio = OUT / "test-audio.wav"
    with wave.open(str(audio), "rb") as stream:
        audio_seconds = stream.getnframes() / stream.getframerate()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    arguments = [
        str(SERVER), "--model", str(MODEL_DIR / "gemma-4-E4B_q4_0-it.gguf"),
        "--mmproj", str(MODEL_DIR / "gemma-4-E4B-it-mmproj.gguf"),
        "--device", "Vulkan0", "--gpu-layers", "all", "--ctx-size", "4096",
        "--parallel", "1", "--threads", "8", "--threads-batch", "8",
        "--host", "127.0.0.1", "--port", str(port), "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}',
    ]
    report = {
        "runtime": "llama.cpp b11669 / 620d2225c / Vulkan",
        "model": str(MODEL_DIR / "gemma-4-E4B_q4_0-it.gguf"),
        "projector": str(MODEL_DIR / "gemma-4-E4B-it-mmproj.gguf"),
        "context_tokens": 4096, "thinking": False,
        "test_image": str(fixture), "audio_seconds": audio_seconds,
        "tests": [], "status": "starting", "server_arguments": arguments,
    }
    save(OUT / "results.json", report)
    started = time.monotonic()
    with (OUT / "server.stdout.log").open("w", encoding="utf-8") as stdout, \
            (OUT / "server.stderr.log").open("w", encoding="utf-8") as stderr:
        server = subprocess.Popen(arguments, cwd=SERVER.parent, stdout=stdout, stderr=stderr,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        report["server_pid"] = server.pid
        save(OUT / "results.json", report)
        print(f"Starting Vulkan server on localhost:{port}; PID {server.pid}", flush=True)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=240, trust_env=False) as client:
                while time.monotonic() - started < 300:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited during loading: {server.returncode}")
                    try:
                        response = client.get("/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("Server did not become healthy within 300 seconds")
                report["load_seconds"] = round(time.monotonic() - started, 3)
                report["status"] = "testing"
                save(OUT / "results.json", report)
                print(f"Model and multimodal projector ready after {report['load_seconds']} seconds", flush=True)
                cases = [
                    {
                        "id": "text", "expected": ", ".join(str(i) for i in range(1, 21)),
                        "content": "Escribe exactamente los números del 1 al 20 separados por comas. No añadas explicaciones.",
                    },
                    {
                        "id": "image", "expected": "5831; de izquierda a derecha: círculo azul, triángulo amarillo, cuadrado rojo",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fixture.read_bytes()).decode()}},
                            {"type": "text", "text": "Responde en español. Transcribe el código numérico de arriba y describe las figuras de izquierda a derecha, indicando forma y color."},
                        ],
                    },
                    {
                        "id": "audio", "expected": "El código de prueba es cuarenta y dos.",
                        "content": [
                            {"type": "text", "text": "Transcribe literalmente lo que se dice en este audio, sin explicaciones."},
                            {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "wav"}},
                        ],
                    },
                ]
                for case in cases:
                    print(f"Running {case['id']} test...", flush=True)
                    request_started = time.monotonic()
                    response = client.post("/v1/chat/completions", json={
                        "messages": [{"role": "user", "content": case["content"]}],
                        "temperature": 0, "seed": 123, "max_tokens": 160,
                        "stream": False, "chat_template_kwargs": {"enable_thinking": False},
                    })
                    elapsed = round(time.monotonic() - request_started, 3)
                    result = {"id": case["id"], "expected": case["expected"], "http_status": response.status_code,
                              "wall_seconds": elapsed}
                    if response.status_code == 200:
                        data = response.json()
                        save(OUT / f"response-{case['id']}.json", data)
                        result.update({"output": data["choices"][0]["message"].get("content"),
                                       "finish_reason": data["choices"][0].get("finish_reason"),
                                       "usage": data.get("usage"), "timings": data.get("timings")})
                    else:
                        result["error"] = response.text[:2000]
                    result["passed"] = assess(result)
                    report["tests"].append(result)
                    save(OUT / "results.json", report)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                report["passed"] = all(result["passed"] for result in report["tests"])
                report["status"] = "complete" if report["passed"] else "completed_with_failures"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            report["server_stopped"] = server.poll() is not None
            save(OUT / "results.json", report)
            print(f"Server stopped. Results: {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
