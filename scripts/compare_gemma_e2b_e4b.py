"""Paired local comparison of E2B/E4B with identical real-media requests."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import benchmark_multimodal_resources as resources
from benchmark_real_multimodal import CpuAccounting, dataset as media_cases
from jev_qat.adapter import JevQatAdapter
from jev_qat.config import RUNTIME_STATE as E4B_STATE, save_json, settings as e4b_settings
from jev_qat.e2b import E2BQatAdapter, RUNTIME_STATE as E2B_STATE, settings as e2b_settings

OUT = ROOT / "inference" / "gemma-e2b-vs-e4b"


def worker_main(model):
    adapter = E2BQatAdapter() if model == "e2b" else JevQatAdapter()
    if model == "e4b":
        native = adapter._post

        def no_cache(route, payload):
            if route == "/v1/chat/completions":
                payload = {**payload, "cache_prompt": False}
            return native(route, payload)

        adapter._post = no_cache
    with adapter:
        print(json.dumps({"ready": True, "model": model}), flush=True)
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if model == "e2b":
                    payload["cache_prompt"] = False
                result = adapter.predict(**payload)
                result["metadata"]["comparison_cache_prompt"] = False
                print(json.dumps({"result": result}, ensure_ascii=True), flush=True)
            except Exception as error:
                print(json.dumps({"error": repr(error)}), flush=True)


class Worker:
    def __init__(self, model):
        self.log = (OUT / f"{model}-worker.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", model], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        if json.loads(self.process.stdout.readline()) != {"ready": True, "model": model}:
            raise RuntimeError("Comparison worker failed to start")

    def predict(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=15)
        self.log.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(report):
    rows = []
    for case in report["cases"]:
        row = {"case": case["id"], "label": case["label"], "expected_source_description": case["expected"]}
        for model in ("e4b", "e2b"):
            trials = [t for t in report["trials"] if t["model"] == model and t["case"] == case["id"] and t["result"]]
            warm = [t for t in trials if t["round"] > 0]
            row[model] = {
                "requests": len(trials), "first_seconds": trials[0]["wall_seconds"] if trials else None,
                "median_seconds": statistics.median(t["wall_seconds"] for t in warm) if warm else None,
                "min_seconds": min((t["wall_seconds"] for t in warm), default=None),
                "max_seconds": max((t["wall_seconds"] for t in warm), default=None),
                "cpu_mean_percent": statistics.mean(t["cpu_total_mean_percent"] for t in warm) if warm else None,
                "gpu_mean_percent": statistics.mean(t["resources"]["gpu_mean_percent"] for t in warm) if warm else None,
                "gpu_peak_percent": max((t["resources"]["gpu_peak_percent"] for t in trials), default=None),
                "rss_peak_gib": max((t["resources"]["rss_peak_gib"] for t in trials), default=None),
                "commit_peak_gib": max((t["resources"]["committed_peak_gib"] for t in trials), default=None),
                "gpu_shared_peak_gib": max((t["resources"]["gpu_shared_peak_gib"] for t in trials), default=None),
                "predictions": [t["result"]["prediction"] for t in trials],
                "content_matches": sum(t["content_matches"] for t in trials),
                "prompt_tokens": sorted({t["result"]["metadata"]["usage"]["prompt_tokens"] for t in trials}),
            }
        row["e2b_speed_ratio"] = row["e4b"]["median_seconds"] / row["e2b"]["median_seconds"] if row["e2b"]["median_seconds"] else None
        rows.append(row)
    return rows


def render(report):
    rows = aggregate(report)
    report["comparison"] = rows
    summary = {"checks": report["checks"], "comparison": rows, "model_files": report["model_files"],
               "both_backends_loaded": True, "simultaneous_inference_requests": 1,
               "independent_cases": 10, "trials": len(report["trials"]),
               "calibration": "Neither model is calibrated in this comparison",
               "content_matches": {m: sum(t["content_matches"] for t in report["trials"] if t["model"] == m) for m in ("e4b", "e2b")}}
    save_json(OUT / "results.json", report)
    save_json(OUT / "summary.json", summary)
    columns = ["case", "model", "first_seconds", "median_seconds", "cpu_mean_percent", "gpu_mean_percent", "rss_peak_gib", "commit_peak_gib", "gpu_shared_peak_gib", "content_matches"]
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for model in ("e4b", "e2b"):
                writer.writerow({"case": row["case"], "model": model, **{k: row[model][k] for k in columns[2:]}})
    lines = ["# Gemma E2B frente a E4B: implementación y comparación local", "",
             "QAT Q4_0 oficial con proyector propio en ambos modelos. Misma versión llama.cpp b11669 Vulkan, "
             "4096 tokens de contexto y ocho hilos. Ambos backends cargados, una sola inferencia simultánea; "
             "orden de modelos alternado por caso y ronda. Diez entradas distintas, cuatro pasadas por modelo: 80 inferencias. "
             "Los tiempos son medianas de las tres pasadas posteriores a la primera; los picos abarcan las cuatro. "
             "No se reutilizan tokens de entrada. Las dos comparaciones usan probabilidades sin calibrar.", "",
             "| Entrada | E4B (s) | E2B (s) | Razón E4B/E2B | CPU media E4B/E2B (%) | GPU media E4B/E2B (%) | RAM residente pico E4B/E2B (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        a, b = row["e4b"], row["e2b"]
        lines.append(f"| {row['label']} | {a['median_seconds']:.3f} | {b['median_seconds']:.3f} | {row['e2b_speed_ratio']:.2f}× | "
                     f"{a['cpu_mean_percent']:.1f} / {b['cpu_mean_percent']:.1f} | {a['gpu_mean_percent']:.1f} / {b['gpu_mean_percent']:.1f} | "
                     f"{a['rss_peak_gib']:.2f} / {b['rss_peak_gib']:.2f} |")
    lines += ["", "Una razón mayor que 1 indica menor latencia con E2B. Son mediciones descriptivas en este equipo, "
              "con otras aplicaciones en ejecución; no son una prueba de significación estadística de calidad.", "",
              "## Memoria y archivos", "",
              "| Modelo | Disco: modelo + proyector (GiB) | GPU compartida pico (GiB) | Privada comprometida pico (GiB) | RAM residente pico (GiB) |",
              "|---|---:|---:|---:|---:|"]
    for model in ("e4b", "e2b"):
        lines.append(f"| {model.upper()} | {sum(x['bytes'] for x in report['model_files'][model])/1024**3:.2f} | "
                     f"{max(r[model]['gpu_shared_peak_gib'] for r in rows):.2f} | "
                     f"{max(r[model]['commit_peak_gib'] for r in rows):.2f} | {max(r[model]['rss_peak_gib'] for r in rows):.2f} |")
    lines += ["", "CPU incluye backend, cliente Python y procesos de conversión/extracción, incluidos los ya terminados, "
              "usando contabilidad Job Object. Porcentaje sobre 16 procesadores lógicos. GPU: motor más ocupado atribuible "
              "a los procesos del modelo. RAM residente suma conjuntos de trabajo; comprometida es memoria privada virtual. "
              "La GPU integrada utiliza RAM compartida y estas cifras se solapan: no sumarlas. Los buffers persisten y el orden "
              "afecta a las reservas. Se mide cada backend y su cliente, excluyendo los puentes HTTP inactivos. "
              "El consumo global del equipo contiene ambos modelos cargados.", "",
              "## Respuestas y controles de contenido", "",
              "Las descripciones esperadas provienen de las fuentes públicas y la inspección visual; no son etiquetas "
              "de un corpus de precisión con adjudicación humana independiente. Cuatro repeticiones de una misma entrada "
              "no constituyen cuatro ejemplos independientes. Las probabilidades de E2B requieren su propia calibración. "
              "El perfil de recuperación de texto v2 se comparte como prompt; su temperatura y rescates específicos de E4B no se transfieren.", "",
              "| Entrada | Descripción esperada | Respuestas E4B | Respuestas E2B | Coincidencias E4B/E2B de 4 |",
              "|---|---|---|---|---:|"]
    for row in rows:
        a, b = row["e4b"], row["e2b"]
        lines.append(f"| {row['label']} | {row['expected_source_description']} | {'; '.join(sorted(set(a['predictions'])))} | "
                     f"{' ; '.join(sorted(set(b['predictions'])))} | {a['content_matches']} / {b['content_matches']} |")
    lines += ["", "Fotos originales de 17,4 y 45,9 MP; dos grabaciones humanas españolas de 15 y 30 s; vídeos reales 1080p "
              "de 30 y 60 s. El adaptador muestrea 2 o 16 fotogramas, reducidos a 640 píxeles de ancho, e ignora la pista sonora del vídeo. "
              "Cada salida es una etiqueta de un token. Procedencia y licencias en inference/real-media-benchmark-20261017/ATTRIBUTION.md.", "",
              "## Uso de E2B", "",
              "API independiente: http://127.0.0.1:18167/docs · POST /predict · Estado: GET /health. "
              "Mismos campos state, question, options, modality, media y video_frames; campo adicional cache_prompt. "
              "Python: from jev_qat.e2b import load_jev_omni_e2b_qat, classify_spanish_e2b. "
              "Gestión: python -m jev_qat.e2b start | status | stop. El servicio E4B y el perfil activo v2 permanecen sin cambios.", "",
              f"Comprobaciones: {json.dumps(report['checks'], ensure_ascii=False)}", "",
              "Repositorios oficiales: [E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf), "
              "[E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf). "
              "Versiones, SHA256 de archivos, peticiones y todas las muestras están en results.json, plan.json y samples-*.jsonl."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    parts, table = [], False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not table:
                parts.append("<div class='scroll'><table>")
                table = True
            parts.append("<tr>" + "".join(f"<td>{html.escape(v.strip())}</td>" for v in line.strip("|").split("|")) + "</tr>")
            continue
        if table:
            parts.append("</table></div>")
            table = False
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line:
            parts.append(f"<p>{html.escape(line)}</p>")
    css = "body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;background:#f5f7fb;color:#172438}p{line-height:1.65}table{border-collapse:collapse;background:white;width:100%}td{border:1px solid #cdd6e3;padding:11px;text-align:right}td:first-child{text-align:left}tr:first-child{font-weight:650;background:#e9eff9}.scroll{overflow:auto}h2{margin-top:36px}"
    (OUT / "report.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>E2B frente a E4B</title><style>' + css + "</style>" + "".join(parts) + "</html>", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=False)
    cases = [c for c in media_cases() if c["kind"] == "real_media" or c["id"] in ("text_v2", "text_short")]
    for case in cases:
        case["payload"].pop("calibration_profile", None)
    configurations = {"e4b": e4b_settings(), "e2b": e2b_settings()}
    runtime_states = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in (("e4b", E4B_STATE), ("e2b", E2B_STATE))}
    frozen_paths = [ROOT / p for p in ("jev_qat_config.json", "classification/current.json", "jev_qat/adapter.py", "jev_qat_e2b_config.json", "jev_qat/e2b.py")]
    frozen = {str(p): sha(p) for p in frozen_paths}
    receipts = {
        "e4b": ROOT / "compatibility" / "qat-q4" / "download-receipt.json",
        "e2b": ROOT / "compatibility" / "e2b-qat-q4" / "download-receipt.json",
    }
    files = {}
    for model, config in configurations.items():
        receipt = json.loads(receipts[model].read_text(encoding="utf-8"))
        files[model] = []
        for field in ("model", "projector"):
            path = Path(config[field])
            verified = receipt["files"][path.name]
            if path.stat().st_size != verified["size"] or not verified["sha256_verified"]:
                raise RuntimeError("Model download has not been verified")
            files[model].append({"path": str(path), "bytes": path.stat().st_size, "sha256": verified["sha256"],
                                 "repository": receipt["model"], "revision": receipt["revision"]})
    report = {"cases": cases, "configuration": configurations, "model_files": files,
              "frozen_files": frozen, "trials": [], "both_models_loaded": True,
              "simultaneous_inferences": 1, "comparison_calibration": "none for either model",
              "backend_pids": {m: s["backend_pid"] for m, s in runtime_states.items()}}
    save_json(OUT / "plan.json", report)
    workers, jobs, monitors = {}, {}, {}
    try:
        for model in ("e4b", "e2b"):
            state = runtime_states[model]
            command = psutil.Process(state["backend_pid"]).cmdline()
            if configurations[model]["model"] not in command:
                raise RuntimeError("Loaded backend does not match this model configuration")
            workers[model] = Worker(model)
            jobs[model] = CpuAccounting(workers[model].process.pid, state["backend_pid"])
            warm = time.perf_counter()
            workers[model].predict(cases[0]["payload"])
            report.setdefault("initialization_seconds_excluded", {})[model] = time.perf_counter() - warm
            monitors[model] = resources.Monitor(workers[model].process.pid, state["backend_pid"], .1)
            monitors[model].start()
        time.sleep(2)
        for round_index in range(4):
            order = cases if round_index % 2 == 0 else list(reversed(cases))
            for case in order:
                position = next(i for i, c in enumerate(cases) if c["id"] == case["id"])
                models = ("e4b", "e2b") if (position + round_index) % 2 == 0 else ("e2b", "e4b")
                for model in models:
                    time.sleep(.3)
                    before = jobs[model].read()
                    started = time.perf_counter()
                    result = error = None
                    try:
                        result = workers[model].predict(case["payload"])
                    except Exception as failure:
                        error = repr(failure)
                    ended = time.perf_counter()
                    after = jobs[model].read()
                    trial = {"model": model, "case": case["id"], "round": round_index,
                             "start": started, "end": ended, "wall_seconds": ended - started,
                             "cpu_job_before": before, "cpu_job_after": after,
                             "cpu_total_mean_percent": 100 * (after["seconds"] - before["seconds"]) / (ended-started) / psutil.cpu_count(),
                             "result": result, "error": error,
                             "content_matches": bool(result and result["prediction"] == case["expected"])}
                    report["trials"].append(trial)
                    save_json(OUT / "progress.json", report)
                    print(json.dumps({"model": model, "case": case["id"], "round": round_index,
                                      "seconds": round(ended-started, 3), "content_matches": trial["content_matches"], "error": error,
                                      "completed": len(report["trials"])}, ensure_ascii=True), flush=True)
        time.sleep(.8)
    finally:
        for model, monitor in monitors.items():
            monitor.close()
            with (OUT / f"samples-{model}.jsonl").open("w", encoding="utf-8") as f:
                for sample in monitor.samples:
                    f.write(json.dumps(sample) + "\n")
        for worker in workers.values():
            worker.close()
        for job in jobs.values():
            job.close()
    for trial in report["trials"]:
        trial["resources"] = monitors[trial["model"]].summary(trial["start"], trial["end"])
    successful = [t for t in report["trials"] if t["result"]]
    report["checks"] = {
        "all_80_inferences_succeeded": len(successful) == 80 and len(report["trials"]) == 80,
        "zero_cached_input_tokens": all(t["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 for t in successful),
        "one_output_token": all(t["result"]["metadata"]["usage"]["completion_tokens"] == 1 for t in successful),
        "both_uncalibrated": all(t["result"]["metadata"]["calibrated"] is False for t in successful),
        "complete_trace_coverage": all(abs(t["resources"]["covered_seconds"] - t["wall_seconds"]) < 1e-6 for t in report["trials"]),
        "profiles_and_source_unchanged": all(sha(Path(p)) == value for p, value in frozen.items()),
        "model_identities_correct": all(t["result"]["metadata"]["backend"] == f"gemma-4-{t['model'].upper()}-it-qat-q4_0-gguf" for t in successful),
    }
    report["sample_intervals"] = {m: {"median_seconds": statistics.median(s["end"]-s["start"] for s in monitor.samples),
                                             "max_seconds": max(s["end"]-s["start"] for s in monitor.samples)} for m, monitor in monitors.items()}
    for model, config in configurations.items():
        with httpx.Client(base_url=config["api_url"], timeout=10, trust_env=False) as client:
            report.setdefault("final_health", {})[model] = client.get("/health").json()
    for name in ("compare_gemma_e2b_e4b.py", "benchmark_real_multimodal.py", "benchmark_multimodal_resources.py"):
        shutil.copy2(ROOT / "scripts" / name, OUT / name)
    render(report)
    print(json.dumps({"checks": report["checks"], "report": str(OUT / "report.md")}), flush=True)
    if not all(report["checks"].values()):
        raise SystemExit("Comparison completed with a failed verification check")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2])
    else:
        main()
