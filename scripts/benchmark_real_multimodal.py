"""Real public media, no prefix cache, same backend and fresh legacy controls."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import httpx
import psutil
import win32api
import win32con
import win32job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_multimodal_resources as previous
from jev_qat.config import RUNTIME_STATE, save_json, settings

OUT = ROOT / "inference" / "real-media-benchmark-20261017"


class CpuAccounting:
    """Counts even exited ffmpeg children, without imposing any job limits."""
    def __init__(self, worker_pid, backend_pid):
        self.job = win32job.CreateJobObject(None, "")
        process = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, worker_pid)
        try:
            win32job.AssignProcessToJobObject(self.job, process)
        finally:
            process.Close()
        self.backend = psutil.Process(backend_pid)

    def read(self):
        info = win32job.QueryInformationJobObject(self.job, win32job.JobObjectBasicAccountingInformation)
        cpu = self.backend.cpu_times()
        return {"seconds": (info["TotalUserTime"] + info["TotalKernelTime"]) / 1e7 + cpu.user + cpu.system,
                "job_processes": info["TotalProcesses"], "job_active_processes": info["ActiveProcesses"]}

    def close(self):
        self.job.Close()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset():
    rows = [{"id": k, "label": label, "payload": payload, "expected": expected,
             "reference_case": k, "kind": "legacy_control"} for k, label, payload, expected in previous.cases()]
    fixtures = OUT / "fixtures"
    image_question = "¿Qué tipo de entorno predomina en la fotografía?"
    image_options = ["montañas y lagos", "ciudad con una torre", "paisaje submarino"]
    for identifier, file, label, expected in (
        ("image_alps", "alps.jpg", "Foto Alpes 10109×4542 (45,9 MP)", image_options[0]),
        ("image_paris", "paris.jpg", "Foto París 4114×4235 (17,4 MP)", image_options[1]),
    ):
        rows.append({"id": identifier, "label": label, "payload": {"state": "", "question": image_question,
                     "options": image_options, "modality": "image", "media": str(fixtures / file)},
                     "expected": expected, "reference_case": "image", "kind": "real_media"})
    for seconds in (15, 30):
        rows.append({"id": f"audio_{seconds}", "label": f"Voz humana española {seconds} s",
                     "payload": {"state": "", "question": "¿En qué idioma habla la persona del audio?",
                                 "options": ["español", "inglés", "alemán"], "modality": "audio",
                                 "media": str(fixtures / f"speech-{seconds}s.wav")},
                     "expected": "español", "reference_case": "audio", "kind": "real_media"})
    for seconds, file, expected in ((30, "bus-30s.mp4", "tráfico urbano con vehículos"),
                                     (60, "marseille-60s.mp4", "vista aérea de ciudad y costa")):
        for frames in (2, 16):
            rows.append({"id": f"video_{seconds}_{frames}", "label": f"Vídeo real {seconds} s / {frames} fotogramas",
                         "payload": {"state": "", "question": "¿Qué tipo de escena predomina en el vídeo?",
                                     "options": ["tráfico urbano con vehículos", "vista aérea de ciudad y costa", "interior de una cocina"],
                                     "modality": "video", "media": str(fixtures / file), "video_frames": frames},
                         "expected": expected, "reference_case": f"video_{frames}", "kind": "real_media"})
    return rows


def aggregate(report):
    old = {r["case"]: r for r in report["historical"]["comparison"]}
    rows = []
    for case in report["cases"]:
        trials = [t for t in report["trials"] if t["case"] == case["id"] and t.get("result")]
        warm = [t for t in trials if t["round"] > 0]
        if not warm:
            continue
        row = {"case": case["id"], "label": case["label"], "kind": case["kind"], "reference_case": case["reference_case"],
               "successful_requests": len(trials), "first_seconds": trials[0]["wall_seconds"],
               "median_seconds": statistics.median(t["wall_seconds"] for t in warm),
               "min_seconds": min(t["wall_seconds"] for t in warm), "max_seconds": max(t["wall_seconds"] for t in warm),
               "cpu_root_mean_percent": statistics.mean(t["cpu_root_mean_percent"] for t in warm),
               "cpu_total_mean_percent": statistics.mean(t["cpu_total_mean_percent"] for t in warm),
               "cpu_sampled_peak_percent": max(t["resources"]["cpu_sampled_peak_percent"] for t in trials),
               "gpu_mean_percent": statistics.mean(t["resources"]["gpu_mean_percent"] for t in warm),
               "gpu_peak_percent": max(t["resources"]["gpu_peak_percent"] for t in trials),
               "rss_peak_gib": max(t["resources"]["rss_peak_gib"] for t in trials),
               "committed_peak_gib": max(t["resources"]["committed_peak_gib"] for t in trials),
               "gpu_shared_peak_gib": max(t["resources"]["gpu_shared_peak_gib"] for t in trials),
               "prompt_tokens": sorted({t["result"]["metadata"]["usage"]["prompt_tokens"] for t in trials}),
               "source_content_check_matches": sum(t["content_check_matches"] for t in trials)}
        historical = old[case["reference_case"]]
        row["historical_seconds"] = historical["warm_median_seconds"]
        row["historical_ratio"] = row["median_seconds"] / historical["warm_median_seconds"]
        row["historical_rss_peak_gib"] = historical["rss_peak_gib"]
        row["historical_cpu_root_mean_percent"] = historical["cpu_mean_percent"]
        row["historical_gpu_mean_percent"] = historical["gpu_mean_percent"]
        rows.append(row)
    controls = {r["case"]: r for r in rows if r["kind"] == "legacy_control"}
    for row in rows:
        reference = controls[row["reference_case"]]
        row["fresh_control_seconds"] = reference["median_seconds"]
        row["fresh_control_ratio"] = row["median_seconds"] / reference["median_seconds"]
    return rows


def render(run, report):
    rows = aggregate(report)
    report["comparison"] = rows
    save_json(run / "results.json", report)
    save_json(OUT / "comparison.json", {"run": str(run), "checks": report["checks"], "comparison": rows})
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Consumo con fotografías y grabaciones reales", "",
             "Mismo Gemma E4B QAT Q4, proyector multimodal, Vulkan, contexto 4096 y ocho hilos. "
             "Caché de entrada desactivada en todas las peticiones, modelo ya cargado. "
             "Cuatro pasadas por caso: se muestran medianas de las tres posteriores a la primera; picos de las cuatro. "
             "Una petición simultánea. Adaptador Python aislado y backend existente, igual que el control sin caché anterior.", "",
             "## Nuevos medios reales", "",
             "| Entrada | Tiempo (s) | Anterior breve (s) | Control breve actual (s) | CPU total media (%) | GPU media / pico (%) | RAM residente pico (GiB) | Comprometida pico (GiB) | GPU compartida pico (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["kind"] != "real_media":
            continue
        lines.append(f"| {row['label']} | {row['median_seconds']:.3f} | {row['historical_seconds']:.3f} | {row['fresh_control_seconds']:.3f} | "
                     f"{row['cpu_total_mean_percent']:.1f} | {row['gpu_mean_percent']:.1f} / {row['gpu_peak_percent']:.1f} | "
                     f"{row['rss_peak_gib']:.2f} | {row['committed_peak_gib']:.2f} | {row['gpu_shared_peak_gib']:.2f} |")
    lines += ["", "## Controles breves repetidos en la misma sesión", "",
              "| Entrada | Histórico (s) | Actual (s) | CPU principal media (%) | CPU con hijos media (%) | GPU media (%) | RAM residente pico (GiB) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["kind"] != "legacy_control":
            continue
        lines.append(f"| {row['label']} | {row['historical_seconds']:.3f} | {row['median_seconds']:.3f} | "
                     f"{row['cpu_root_mean_percent']:.1f} | {row['cpu_total_mean_percent']:.1f} | {row['gpu_mean_percent']:.1f} | {row['rss_peak_gib']:.2f} |")
    lines += ["", "## Cómo interpretar las cifras", "",
              "La CPU está normalizada sobre los 16 procesadores lógicos. La nueva columna CPU total suma backend, cliente y "
              "todos los procesos ffmpeg/ffprobe, incluidos los ya terminados, usando contabilidad Job Object de Windows. "
              "Se conserva CPU principal para comparar con la medición histórica que no incluía CPU de hijos en su media.", "",
              "GPU representa el motor más ocupado atribuible a los procesos observados en cada intervalo; pico muestreado. "
              "RAM residente suma conjuntos de trabajo; memoria comprometida representa memoria privada virtual. "
              "La GPU integrada usa RAM compartida: estas cifras se solapan y no se deben sumar. Los buffers del backend "
              "persisten y el orden puede influir en la memoria; no son incrementos aislados de cada modalidad.", "",
              "Se descargaron originales y se prepararon recortes explícitos antes del cronómetro. La latencia medida incluye "
              "lectura, codificación base64, conversión a audio mono 16 kHz y extracción de imágenes del vídeo. "
              "Las fotografías se enviaron con sus bytes y resolución originales; el codificador del modelo limita la resolución interna "
              "según su presupuesto visual. El adaptador reduce los fotogramas de vídeo a 640 píxeles de ancho.", "",
              "Los audios de 15 y 30 s se procesan completos y cumplen el máximo de 30 s. El vídeo se evalúa mediante "
              "2 o 16 fotogramas distribuidos sobre los 30 o 60 s completos, sin su pista sonora. Duración mayor no implica "
              "más fotogramas ni más tokens visuales si se mantiene ese número. Cada respuesta sigue siendo una etiqueta de un token.", "",
              "Las descripciones de contenido se contrastan con las fuentes y una inspección visual; no es un corpus "
              "de precisión etiquetado por humanos ni una evaluación estadística de clasificación. El objetivo es consumo y latencia. "
              "El equipo siguió ejecutando otras aplicaciones. Primeras pasadas, rangos, tokens y datos completos están en results.json.", "",
              f"Comprobaciones: {json.dumps(report['checks'], ensure_ascii=False)}", "",
              f"Muestreo real: {json.dumps(report['sample_intervals'], ensure_ascii=False)}", "",
              "## Fuentes y licencias", ""]
    for source in report["media_manifest"]["sources"]:
        lines.append(f"- [{source['title']}]({source['source_page']}), {source['author']}; [{source['license']}]({source['license_url']}).")
    lines += ["", "Manifest de archivos, SHA256, recortes y comandos: media-manifest.json. Atribución: ATTRIBUTION.md. "
              "No se modificó el perfil activo v2 ni la configuración persistente del servicio.", "",
              "Método de CPU: [Job Object de Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    parts, table = [], False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not table:
                parts.append("<div class='scroll'><table>")
                table = True
            parts.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in line.strip("|").split("|")) + "</tr>")
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
    style = "body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;background:#f5f7fb;color:#162338}p{line-height:1.6}table{border-collapse:collapse;width:100%;background:white}td{padding:11px;border:1px solid #cdd6e2;text-align:right}td:first-child{text-align:left}tr:first-child{font-weight:650;background:#e8eef8}.scroll{overflow:auto}h2{margin-top:36px}"
    (OUT / "report.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Consumo con medios reales</title><style>' + style + "</style>" + "".join(parts) + "</html>", encoding="utf-8")


def main():
    run = OUT / "run-01"
    run.mkdir(exist_ok=False)
    config = settings()
    state = json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "media-manifest.json").read_text(encoding="utf-8"))
    frozen = {str(p): digest(ROOT / p) for p in (Path("jev_qat_config.json"), Path("classification/current.json"), Path("jev_qat/adapter.py"))}
    cases = dataset()
    historical = json.loads((ROOT / "inference" / "resource-benchmark-20261017-no-cache" / "results.json").read_text(encoding="utf-8"))
    report = {"cases": cases, "media_manifest": manifest, "historical": historical,
              "frozen_source_hashes": frozen, "trials": [], "mode": "adapter_no_cache",
              "service_pid": state["pid"], "backend_pid": state["backend_pid"], "configuration": config,
              "content_checks_are_source_based_not_independent_human_gold": True}
    for file in manifest["fixtures"]:
        assert digest(Path(file["path"])) == file["sha256"]
    worker = previous.Worker(run)
    accounting = CpuAccounting(worker.process.pid, state["backend_pid"])
    warm_start = time.perf_counter()
    worker.predict(cases[0]["payload"])
    report["initialization_excluded_seconds"] = time.perf_counter() - warm_start
    monitor = previous.Monitor(worker.process.pid, state["backend_pid"], .1)
    monitor.start()
    try:
        time.sleep(2)
        for round_index in range(4):
            order = cases if round_index % 2 == 0 else list(reversed(cases))
            for case in order:
                time.sleep(.3)
                root_before = previous.root_cpu(monitor.roots)
                total_before = accounting.read()
                start = time.perf_counter()
                result, error = None, None
                try:
                    result = worker.predict(case["payload"])
                except Exception as failure:
                    error = repr(failure)
                end = time.perf_counter()
                total_after = accounting.read()
                root_after = previous.root_cpu(monitor.roots)
                time.sleep(.25)
                trial = {"case": case["id"], "round": round_index, "start": start, "end": end,
                         "wall_seconds": end - start, "result": result, "error": error,
                         "cpu_root_mean_percent": 100 * (root_after - root_before) / (end - start) / psutil.cpu_count(),
                         "cpu_total_mean_percent": 100 * (total_after["seconds"] - total_before["seconds"]) / (end - start) / psutil.cpu_count(),
                         "cpu_job_before": total_before, "cpu_job_after": total_after,
                         "content_check_matches": bool(result and result["prediction"] == case["expected"])}
                report["trials"].append(trial)
                save_json(run / "progress.json", report)
                print(json.dumps({"round": round_index, "case": case["id"], "seconds": round(end-start, 3),
                                  "content_matches": trial["content_check_matches"], "error": error,
                                  "cache_tokens": result["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] if result else None}, ensure_ascii=True), flush=True)
    finally:
        time.sleep(.3)
        monitor.close()
        worker.close()
        accounting.close()
        with (run / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for sample in monitor.samples:
                stream.write(json.dumps(sample) + "\n")
    for trial in report["trials"]:
        trial["resources"] = monitor.summary(trial["start"], trial["end"])
    intervals = [s["end"] - s["start"] for s in monitor.samples]
    report["sample_intervals"] = {"median_seconds": statistics.median(intervals), "max_seconds": max(intervals), "samples": len(intervals)}
    success = [t for t in report["trials"] if t["result"]]
    report["checks"] = {
        "all_56_requests_succeeded": len(success) == 56 and len(report["trials"]) == 56,
        "zero_cached_input_tokens": all(t["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 for t in success),
        "one_output_token": all(t["result"]["metadata"]["usage"]["completion_tokens"] == 1 for t in success),
        "full_monitor_interval_coverage": all(abs(t["resources"]["covered_seconds"] - t["wall_seconds"]) < 1e-6 for t in report["trials"]),
        "configuration_profile_source_unchanged": all(digest(ROOT / Path(name)) == value for name, value in frozen.items()),
        "job_cpu_totals_valid": all(t["cpu_total_mean_percent"] >= 0 for t in report["trials"]),
        "monitor_no_error": monitor.error is None,
    }
    with httpx.Client(base_url=config["api_url"], timeout=10, trust_env=False) as client:
        report["final_health"] = client.get("/health").json()
    for name in ("benchmark_real_multimodal.py", "benchmark_multimodal_resources.py", "prepare_real_multimodal_benchmark.py"):
        shutil.copy2(ROOT / "scripts" / name, run / name)
    render(run, report)
    print(json.dumps({"checks": report["checks"], "comparison": report["comparison"], "report": str(OUT / "report.md")}, ensure_ascii=True), flush=True)
    if not all(report["checks"].values()):
        raise SystemExit("Recorded benchmark contains a failed verification check")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
