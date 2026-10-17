"""Measure the running Windows bridge without changing model or task settings."""
import argparse
import csv
import hashlib
import html
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil
import win32pdh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_qat.config import RUNTIME_STATE, save_json, settings
from jev_qat.eval_data import CATEGORIES, QUESTION

GIB = 1024**3


class Monitor:
    """PDH GPU intervals plus sampled process working sets and CPU times."""

    def __init__(self, service_pid, backend_pid, interval):
        self.roots = [psutil.Process(service_pid), psutil.Process(backend_pid)]
        self.interval = interval
        self.samples = []
        self.cpu_max = {}
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def processes(self):
        found = {p.pid: p for p in self.roots}
        for root in self.roots:
            for child in root.children(recursive=True):
                found[child.pid] = child
        return found

    def run(self):
        query = None
        try:
            query = win32pdh.OpenQuery()
            paths = {
                "gpu_shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
                "gpu_dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
                "gpu_engines": r"\GPU Engine(*)\Utilization Percentage",
            }
            counters = {k: win32pdh.AddEnglishCounter(query, p) for k, p in paths.items()}
            win32pdh.CollectQueryData(query)
            previous = time.perf_counter()
            psutil.cpu_percent()
            self.ready.set()
            while not self.stop.wait(self.interval):
                procs = self.processes()
                rss = committed = 0
                names = {}
                for pid, process in procs.items():
                    try:
                        with process.oneshot():
                            mem = process.memory_info()
                            cpu = process.cpu_times()
                            names[str(pid)] = process.name()
                        rss += mem.rss
                        committed += mem.private
                        identity = (pid, process.create_time())
                        self.cpu_max[identity] = max(self.cpu_max.get(identity, 0), cpu.user + cpu.system)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                win32pdh.CollectQueryData(query)
                gpu = {}
                prefixes = tuple(f"pid_{pid}_" for pid in procs)
                for key, counter in counters.items():
                    values = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_DOUBLE)
                    matching = {k: max(0.0, v) for k, v in values.items() if k.startswith(prefixes)}
                    if key == "gpu_engines":
                        gpu["gpu_busy_percent"] = min(100.0, max(matching.values(), default=0))
                        gpu["gpu_engine_percent"] = matching
                    else:
                        gpu[key] = sum(matching.values())
                now = time.perf_counter()
                memory = psutil.virtual_memory()
                self.samples.append({
                    "start": previous, "end": now, "rss_bytes": rss,
                    "private_committed_bytes": committed,
                    "cpu_seconds_cumulative": sum(self.cpu_max.values()),
                    "system_cpu_percent": psutil.cpu_percent(),
                    "system_available_bytes": memory.available,
                    "processes": names, **gpu,
                })
                previous = now
        except Exception as error:
            self.error = repr(error)
            self.ready.set()
        finally:
            if query is not None:
                win32pdh.CloseQuery(query)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15) or self.error:
            raise RuntimeError(f"Resource monitor unavailable: {self.error}")

    def close(self):
        self.stop.set()
        self.thread.join(15)
        if self.error or self.thread.is_alive():
            raise RuntimeError(f"Resource monitor failed: {self.error}")

    def summary(self, start, end):
        selected = [s for s in self.samples if s["end"] > start and s["start"] < end]
        if not selected:
            raise RuntimeError("No resource sample overlaps this request")
        weights = [max(0, min(s["end"], end) - max(s["start"], start)) for s in selected]
        weight = sum(weights)
        weighted = lambda key: sum(s[key] * w for s, w in zip(selected, weights)) / weight
        preceding = next((s for s in reversed(self.samples) if s["end"] <= start), selected[0])
        cpu_peaks = []
        previous = preceding
        for sample in selected:
            dt = sample["end"] - previous["end"]
            if dt > 0:
                cpu_peaks.append(max(0, 100 * (sample["cpu_seconds_cumulative"] - previous["cpu_seconds_cumulative"]) / dt / psutil.cpu_count()))
            previous = sample
        return {
            "sample_count": len(selected), "covered_seconds": weight,
            "rss_peak_gib": max(s["rss_bytes"] for s in selected) / GIB,
            "rss_mean_gib": weighted("rss_bytes") / GIB,
            "committed_peak_gib": max(s["private_committed_bytes"] for s in selected) / GIB,
            "gpu_shared_peak_gib": max(s["gpu_shared_bytes"] for s in selected) / GIB,
            "gpu_dedicated_peak_gib": max(s["gpu_dedicated_bytes"] for s in selected) / GIB,
            "gpu_mean_percent": weighted("gpu_busy_percent"),
            "gpu_peak_percent": max(s["gpu_busy_percent"] for s in selected),
            "cpu_sampled_peak_percent": max(cpu_peaks, default=0),
            "system_cpu_mean_percent": weighted("system_cpu_percent"),
            "system_available_min_gib": min(s["system_available_bytes"] for s in selected) / GIB,
            "pre_request_rss_gib": preceding["rss_bytes"] / GIB,
            "pre_request_gpu_shared_gib": preceding["gpu_shared_bytes"] / GIB,
        }


def root_cpu(roots):
    return sum(p.cpu_times().user + p.cpu_times().system for p in roots)


def worker_main():
    """Isolated client; overrides only its own native request cache flag."""
    from jev_qat.adapter import JevQatAdapter
    with JevQatAdapter() as adapter:
        native_post = adapter._post

        def no_prefix_cache(route, payload):
            if route == "/v1/chat/completions":
                payload = {**payload, "cache_prompt": False}
            return native_post(route, payload)

        adapter._post = no_prefix_cache
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            try:
                result = adapter.predict(**json.loads(line))
                result["metadata"]["benchmark_cache_prompt"] = False
                print(json.dumps({"result": result}, ensure_ascii=True), flush=True)
            except Exception as error:
                print(json.dumps({"error": repr(error)}), flush=True)


class Worker:
    def __init__(self, out):
        self.log = (out / "worker-stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, encoding="utf-8",
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        if json.loads(self.process.stdout.readline()) != {"ready": True}:
            raise RuntimeError("Benchmark worker failed to start")

    def predict(self, payload):
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Benchmark worker exited unexpectedly; inspect worker-stderr.log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=15)
        self.log.close()


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,sample_rate,channels,duration,nb_frames:format=duration,size",
                        "-of", "json", str(path)], capture_output=True, check=True, timeout=20,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": json.loads(r.stdout)}


def cases():
    fixtures = ROOT / "inference" / "qat-q4-vulkan"
    text = {"state": "Añade manzanas y leche a mi lista de compras", "question": QUESTION,
            "options": list(CATEGORIES.values()), "task_profile": "massive_es_v2",
            "calibration_profile": "massive_scenario_es_v2"}
    short = {"state": "El código que aparece arriba es 5831.",
             "question": "¿Qué código numérico aparece arriba?", "options": ["8513", "5831", "1385"]}
    image = {"state": "", "question": "¿Qué código numérico aparece arriba en la imagen?",
             "options": ["8513", "5831", "1385"], "modality": "image", "media": str(fixtures / "test-image.png")}
    audio = {"state": "", "question": "¿Qué número se menciona en el audio?",
             "options": ["17", "42", "81"], "modality": "audio", "media": str(fixtures / "test-audio.wav")}
    video = {"state": "", "question": "¿Qué figura aparece en el centro del video?",
             "options": ["círculo azul", "triángulo amarillo", "cuadrado rojo"], "modality": "video",
             "media": str(ROOT / "integration" / "jev-qat" / "test-video.mp4"), "video_frames": 2}
    return [
        ("text_v2", "Texto: clasificador v2", text, CATEGORIES["lists"]),
        ("text_short", "Texto: control de 3 opciones", short, "5831"),
        ("image", "Imagen 640×360", image, "5831"),
        ("audio", "Audio de 3,30 s", audio, "42"),
        ("video_2", "Vídeo de 2 s / 2 fotogramas", video, "triángulo amarillo"),
        ("video_16", "Vídeo de 2 s / 16 fotogramas", {**video, "video_frames": 16}, "triángulo amarillo"),
    ]


def render(out, report):
    rows = []
    for identifier, label, _, _ in cases():
        trials = [t for t in report["trials"] if t["case"] == identifier]
        warm = trials[1:]
        row = {
            "case": identifier, "label": label, "requests": len(trials),
            "correct": sum(t["correct"] for t in trials),
            "first_seconds": trials[0]["wall_seconds"],
            "warm_median_seconds": statistics.median(t["wall_seconds"] for t in warm),
            "warm_min_seconds": min(t["wall_seconds"] for t in warm),
            "warm_max_seconds": max(t["wall_seconds"] for t in warm),
            "cpu_mean_percent": statistics.mean(t["cpu_root_mean_percent"] for t in warm),
            "cpu_peak_percent": max(t["resources"]["cpu_sampled_peak_percent"] for t in trials),
            "gpu_mean_percent": statistics.mean(t["resources"]["gpu_mean_percent"] for t in warm),
            "gpu_peak_percent": max(t["resources"]["gpu_peak_percent"] for t in trials),
            "rss_peak_gib": max(t["resources"]["rss_peak_gib"] for t in trials),
            "committed_peak_gib": max(t["resources"]["committed_peak_gib"] for t in trials),
            "gpu_shared_peak_gib": max(t["resources"]["gpu_shared_peak_gib"] for t in trials),
        }
        rows.append(row)
    report["comparison"] = rows
    save_json(out / "results.json", report)
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    policy = ("caché permitida a través del servicio HTTP" if report["mode"] == "api_cached"
              else "caché de entrada desactivada en un cliente Python aislado que usa el mismo adaptador y backend")
    lines = ["# Recursos de Gemma E4B QAT Q4 multimodal", "",
             f"Cuatro peticiones por caso: primera observada y tres repeticiones idénticas con {policy}. "
             "El servidor ya estaba cargado; no mide arranque en frío. Una sola petición simultánea. "
             "La respuesta del adaptador es una etiqueta de un token, no una conversación larga.", "",
             "| Caso | Primera (s) | Mediana caliente (s) | CPU media / pico (%) | GPU media / pico (%) | RAM residente pico (GiB) | Comprometida pico (GiB) | GPU compartida pico (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['first_seconds']:.3f} | {r['warm_median_seconds']:.3f} | "
                     f"{r['cpu_mean_percent']:.1f} / {r['cpu_peak_percent']:.1f} | "
                     f"{r['gpu_mean_percent']:.1f} / {r['gpu_peak_percent']:.1f} | "
                     f"{r['rss_peak_gib']:.2f} | {r['committed_peak_gib']:.2f} | {r['gpu_shared_peak_gib']:.2f} |")
    lines += ["", "CPU media: procesos principales del servicio y modelo, normalizada sobre los 16 procesadores lógicos. "
              "El pico muestreado incluye los procesos hijos observados (ffmpeg/ffprobe). "
              "GPU: motor más ocupado del servicio en cada intervalo PDH; los picos son muestreados, no máximos garantizados. "
              "El tiempo HTTP incluye lectura, conversión y extracción de fotogramas.", "",
              "La RAM residente corresponde a los conjuntos de trabajo de procesos; la comprometida es memoria privada virtual. "
              "La GPU integrada utiliza RAM compartida. Las tres cifras se solapan y no se deben sumar. "
              "Los procesos hijos de duración inferior al intervalo de muestreo pueden escapar al registro.", "",
              "La imagen y el vídeo son sintéticos. El vídeo repite la misma imagen; 16 fotogramas mide carga visual, "
              "no comprensión temporal ni la calidad con vídeos reales. Una muestra por modalidad no es un benchmark general.", "",
              "La prueba de texto anterior tardó 1,524 s; tenía 741 tokens de entrada, 573 en caché. "
              "La comparación describe estos archivos y estas condiciones de caché; no reproduce necesariamente la carga de la prueba anterior.", "",
              "Los buffers del backend permanecen asignados tras peticiones grandes; la memoria depende también del orden de ejecución. "
              "El pico de una modalidad no es su coste incremental aislado.", "",
              "En el control sin caché, el monitor cuenta el cliente Python de inferencia y el backend, excluyendo el puente HTTP inactivo. "
              "La latencia incluye la misma conversión de medios del adaptador y una comunicación local por tubería. "
              "No se modifican el servicio HTTP, los perfiles ni la configuración persistente.", "",
              f"Verificación: {report['checks']}. Datos completos: results.json; muestras: samples.jsonl; tabla: comparison.csv."]
    markdown = "\n".join(lines) + "\n"
    (out / "report.md").write_text(markdown, encoding="utf-8")
    headers = ["Caso", "Primera s", "Caliente s", "CPU media/pico %", "GPU media/pico %", "RAM GiB", "Comprometida GiB", "GPU compartida GiB"]
    body = []
    for r in rows:
        cells = [r["label"], f"{r['first_seconds']:.3f}", f"{r['warm_median_seconds']:.3f}",
                 f"{r['cpu_mean_percent']:.1f} / {r['cpu_peak_percent']:.1f}",
                 f"{r['gpu_mean_percent']:.1f} / {r['gpu_peak_percent']:.1f}",
                 f"{r['rss_peak_gib']:.2f}", f"{r['committed_peak_gib']:.2f}", f"{r['gpu_shared_peak_gib']:.2f}"]
        body.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in cells) + "</tr>")
    notes = "".join(f"<p>{html.escape(line)}</p>" for line in lines if line and not line.startswith(("|", "#")))
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><title>Consumo multimodal local</title>'
    document += '<style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#162331;background:#f4f7fa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #ccd5df;text-align:right}td:first-child,th:first-child{text-align:left}p{line-height:1.6}</style>'
    document += '<h1>Consumo de Gemma E4B QAT Q4 multimodal</h1><table><thead><tr>'
    document += "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>" + notes + "</html>"
    (out / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.1)
    parser.add_argument("--mode", choices=("api_cached", "adapter_no_cache"), default="api_cached")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = settings()
    # Select PIDs only: never serialize the management token from runtime state.
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
    service_pid, backend_pid = state["pid"], state["backend_pid"]
    active_file = ROOT / "classification" / "current.json"
    active_before = active_file.read_bytes()
    dataset = cases()
    files = sorted({Path(c[2]["media"]) for c in dataset if "media" in c[2]})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "mode": args.mode,
              "python": sys.version, "logical_processors": psutil.cpu_count(),
              "physical_ram_gib": psutil.virtual_memory().total / GIB,
              "service_pid": service_pid, "backend_pid": backend_pid,
              "configuration": {k: config[k] for k in ("device", "context_tokens", "threads", "audio_max_seconds", "max_video_frames")},
              "fixtures": [probe(p) for p in files], "sampling_interval_requested_seconds": args.interval,
              "schedule": [], "trials": [], "previous_text_seconds": 1.5235681533813477,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    worker = Worker(out) if args.mode == "adapter_no_cache" else None
    if worker:
        report["worker_pid"] = worker.process.pid
        # Load the text retrieval bank before timing; keep the model loaded throughout.
        warm_start = time.perf_counter()
        warm = worker.predict(dataset[0][2])
        report["worker_initialization_request"] = {"wall_seconds": time.perf_counter() - warm_start,
                                                   "prediction": warm["prediction"], "metadata": warm["metadata"]}
    monitor = Monitor(worker.process.pid if worker else service_pid, backend_pid, args.interval)
    monitor.start()
    try:
        with httpx.Client(base_url=config["api_url"], timeout=180, trust_env=False) as client:
            health = client.get("/health")
            health.raise_for_status()
            report["health"] = health.json()
            time.sleep(2)
            baseline_end = time.perf_counter()
            report["idle"] = monitor.summary(baseline_end - 1.5, baseline_end)
            for round_index in range(4):
                order = dataset if round_index % 2 == 0 else list(reversed(dataset))
                for identifier, label, payload, expected in order:
                    if monitor.error:
                        raise RuntimeError(monitor.error)
                    report["schedule"].append({"round": round_index, "case": identifier})
                    time.sleep(.3)
                    cpu_before = root_cpu(monitor.roots)
                    start = time.perf_counter()
                    if worker:
                        result = worker.predict(payload)
                    else:
                        response = client.post("/predict", json=payload)
                        response.raise_for_status()
                        result = response.json()
                    end = time.perf_counter()
                    cpu_after = root_cpu(monitor.roots)
                    time.sleep(.22)
                    resources = monitor.summary(start, end)
                    item = {"case": identifier, "round": round_index,
                            "phase": "first_observed" if round_index == 0 else "warm_repeat",
                            "start": start, "end": end, "wall_seconds": end - start,
                            "cpu_root_seconds": cpu_after - cpu_before,
                            "cpu_root_mean_percent": 100 * (cpu_after - cpu_before) / (end - start) / psutil.cpu_count(),
                            "resources": resources, "payload": payload, "expected": expected,
                            "correct": result["prediction"] == expected, "result": result}
                    report["trials"].append(item)
                    save_json(out / "progress.json", report)
                    print(json.dumps({"round": round_index, "case": identifier, "seconds": round(end - start, 3),
                                      "correct": item["correct"], "rss_gib": round(resources["rss_peak_gib"], 2),
                                      "gpu_shared_gib": round(resources["gpu_shared_peak_gib"], 2),
                                      "gpu_peak_percent": round(resources["gpu_peak_percent"], 1)}, ensure_ascii=True), flush=True)
    finally:
        monitor.close()
        if worker:
            worker.close()
        with (out / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in monitor.samples:
                stream.write(json.dumps(row) + "\n")
    steps = [s["end"] - s["start"] for s in monitor.samples]
    # Aggregate from the complete trace: a slow sample may finish after the
    # short post-request pause, while the next request is already starting.
    for trial in report["trials"]:
        trial["resources"] = monitor.summary(trial["start"], trial["end"])
    report["actual_sample_interval_seconds"] = {"median": statistics.median(steps), "max": max(steps), "samples": len(steps)}
    report["checks"] = {
        "all_24_predictions_correct": len(report["trials"]) == 24 and all(t["correct"] for t in report["trials"]),
        "one_output_token": all(t["result"]["metadata"]["usage"]["completion_tokens"] == 1 for t in report["trials"]),
        "requested_video_frames": all(t["result"]["metadata"]["media_items"] == t["payload"]["video_frames"] for t in report["trials"] if t["case"].startswith("video_")),
        "every_request_sampled": all(t["resources"]["sample_count"] > 0 for t in report["trials"]),
        "complete_interval_coverage": all(abs(t["resources"]["covered_seconds"] - t["wall_seconds"]) < 1e-6
                                          for t in report["trials"]),
        "active_profile_unchanged": active_file.read_bytes() == active_before,
        "monitor_no_error": monitor.error is None,
        "cache_policy_verified": (all(t["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] <= 1
                                       for t in report["trials"]) if worker else True),
    }
    render(out, report)
    print(json.dumps({"output": str(out), "checks": report["checks"], "comparison": report["comparison"]}, ensure_ascii=True), flush=True)
    if not all(report["checks"].values()):
        raise SystemExit("Benchmark recorded, but a verification check failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        main()
