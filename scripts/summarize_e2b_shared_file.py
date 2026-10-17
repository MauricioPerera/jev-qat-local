"""Independently verify the shared-file measurements and publish a local report."""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference/e2b-shared-file-benchmark"
GIB = 1024**3
LABELS = {"image_alps": "Imagen 45,9 MP", "audio_30": "Audio de 30 s", "video_60_16": "Vídeo de 60 s / 16 fotogramas"}
MODES = {"shared": "Archivo compartido", "separate": "Envíos separados"}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources(samples, start, end):
    selected = [(s, min(s["end"], end) - max(s["start"], start)) for s in samples
                if s["start"] < end and s["end"] > start]
    covered = sum(w for _, w in selected)
    assert selected and covered > 0

    def mean(key):
        return sum(s[key] * w for s, w in selected) / covered

    def peak(key):
        return max(s[key] for s, _ in selected)

    return {"sample_count": len(selected), "covered_seconds": covered,
            "rss_peak_gib": peak("rss_bytes") / GIB, "rss_mean_gib": mean("rss_bytes") / GIB,
            "committed_peak_gib": peak("private_committed_bytes") / GIB,
            "gpu_shared_peak_gib": peak("gpu_shared_bytes") / GIB,
            "gpu_dedicated_peak_gib": peak("gpu_dedicated_bytes") / GIB,
            "gpu_mean_percent": mean("gpu_busy_percent"), "gpu_peak_percent": peak("gpu_busy_percent"),
            "system_cpu_mean_percent": mean("system_cpu_percent"),
            "system_available_min_gib": min(s["system_available_bytes"] for s, _ in selected) / GIB}


def verify(report, samples):
    checks = {}
    trials = report["trials"]
    cases = {c["id"]: c for c in report["cases"]}
    checks["benchmark_checks"] = all(report["checks"].values())
    checks["24_unique_workflows"] = len(trials) == 24 and len({(t["case"], t["mode"], t["round"]) for t in trials}) == 24
    checks["72_complete_predictions"] = sum(len(t["answers"]) for t in trials) == 72 and all(t["error"] is None and len(t["answers"]) == 3 for t in trials)
    checks["contiguous_raw_sampling"] = bool(samples) and all(abs(a["end"] - b["start"]) < 1e-9 for a, b in zip(samples, samples[1:]))
    checks["valid_raw_samples"] = all(s["end"] > s["start"] and 0 <= s["gpu_busy_percent"] <= 100
                                      and all(math.isfinite(s[k]) and s[k] >= 0 for k in ("rss_bytes", "private_committed_bytes", "gpu_shared_bytes", "gpu_dedicated_bytes")) for s in samples)
    checks["workflows_do_not_overlap"] = all(a["end"] <= b["start"] for a, b in zip(trials, trials[1:]))
    checks["model_and_sources_unchanged"] = all(digest(p) == d for p, d in report["frozen_sources"].items())
    checks["media_unchanged"] = all(digest(c["source"]) == c["sha256"] for c in cases.values())
    checks["archived_measurement_code_matches"] = all(digest(OUT / n) == digest(ROOT / "scripts" / n) for n in ("benchmark_e2b_shared_file.py", "benchmark_multimodal_resources.py"))
    expected_shared = ["read_and_upload", *(["count_tokens"] * 3), *(["predict"] * 3), "remove_copy"]
    expected_separate = ["read_and_upload", "count_tokens", "predict", "remove_copy"] * 3
    checks["workflow_order_and_upload_bytes"] = all(
        [p["kind"] for p in t["phases"]] == (expected_shared if t["mode"] == "shared" else expected_separate)
        and len(t["uploads"]) == (1 if t["mode"] == "shared" else 3)
        and t["upload_bytes"] == cases[t["case"]]["bytes"] * len(t["uploads"])
        and t["question_order"] == [(i + t["round"]) % 3 for i in range(3)]
        and [a["question_id"] for a in t["answers"]] == t["question_order"] for t in trials)
    checks["cpu_accounting_formula"] = all(math.isclose(t["cpu_seconds"], t["cpu_after"]["seconds"] - t["cpu_before"]["seconds"], abs_tol=1e-9)
        and math.isclose(t["cpu_mean_percent"], 100 * t["cpu_seconds"] / t["wall_seconds"] / report["logical_processors"], abs_tol=1e-9)
        and t["cpu_after"]["job_processes"] >= t["cpu_before"]["job_processes"] for t in trials)
    checks["codec_processes_accounted_by_job"] = all(t["cpu_after"]["job_processes"] > t["cpu_before"]["job_processes"]
                                                   for t in trials if t["case"] in ("audio_30", "video_60_16"))
    checks["timestamps_consistent"] = all(math.isclose(t["wall_seconds"], t["end"] - t["start"], abs_tol=1e-9)
        and 0 < t["first_response_seconds"] <= t["last_response_seconds"] <= t["wall_seconds"]
        and all(t["start"] <= p["start"] <= p["end"] <= t["end"] and math.isclose(p["seconds"], p["end"] - p["start"], abs_tol=1e-9) for p in t["phases"])
        and all(a["end"] <= b["start"] for a, b in zip(t["phases"], t["phases"][1:])) for t in trials)
    no_cache = budgets = vectors = backend = True
    counts = {}
    for t in trials:
        for a in t["answers"]:
            meta = a["result"]["metadata"]
            usage = meta["usage"]
            no_cache &= usage["prompt_tokens_details"]["cached_tokens"] == meta["timings"]["cache_n"] == 0 and meta["cache_prompt"] is False
            budgets &= a["budget"]["fits_context"] is True and a["budget"]["inference_executed"] is False and usage["completion_tokens"] == 1 and usage["prompt_tokens"] == a["budget"]["input_tokens"]
            counts.setdefault((t["case"], a["question_id"]), set()).add(usage["prompt_tokens"])
            p = a["result"]["probabilities"]
            opts = cases[t["case"]]["questions"][a["question_id"]]["options"]
            vectors &= set(p) == set(opts) and all(math.isfinite(x) and 0 <= x <= 1 for x in p.values()) and abs(sum(p.values()) - 1) < 1e-6
            backend &= meta["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf"
    checks.update({"zero_prompt_cache_for_all_72": no_cache, "exact_preflight_counts": budgets,
                   "identical_tokens_for_each_paired_question": all(len(x) == 1 for x in counts.values()),
                   "independent_normalized_probability_vectors": vectors, "correct_e2b_backend": backend})
    recalculated = []
    for t in trials:
        for interval in [t, *t["phases"]]:
            r = resources(samples, interval["start"], interval["end"])
            recalculated.append(abs(r["covered_seconds"] - (interval["end"] - interval["start"])) < 1e-6
                                and all(math.isclose(value, interval["resources"][key], abs_tol=1e-8) for key, value in r.items()))
    checks["all_intervals_recomputed_from_raw_trace"] = all(recalculated)
    ids = [u["upload_id"] for t in trials for u in t["uploads"]]
    uploads = ROOT / "integration/jev-qat-e2b/uploads"
    checks["48_unique_uploads_removed"] = len(ids) == len(set(ids)) == 48 and all(not list(uploads.glob(identifier + ".*")) for identifier in ids)
    with httpx.Client(timeout=10, trust_env=False) as client:
        checks["e2b_and_e4b_services_healthy_after_test"] = all(client.get(f"http://127.0.0.1:{port}/health").status_code == 200 for port in (18167, 18165))
    checks["18_steady_workflows_54_predictions"] = len([t for t in trials if t["round"] > 0]) == 18 and sum(len(t["answers"]) for t in trials if t["round"] > 0) == 54
    return {"checks": checks, "all_passed": all(checks.values()), "raw_samples": len(samples),
            "prediction_count": 72, "warmup_predictions_excluded": 18,
            "tokens_by_case_and_question": {f"{c}/Q{q+1}": sorted(v) for (c, q), v in counts.items()}}


def summarize(report):
    rows = []
    comparisons = []
    for case in report["cases"]:
        for mode in MODES:
            ts = [t for t in report["trials"] if t["round"] > 0 and t["case"] == case["id"] and t["mode"] == mode]
            row = {"case": case["id"], "label": LABELS[case["id"]], "mode": mode, "repeats": len(ts),
                   "wall_seconds": statistics.median(t["wall_seconds"] for t in ts),
                   "min_seconds": min(t["wall_seconds"] for t in ts), "max_seconds": max(t["wall_seconds"] for t in ts),
                   "first_response_seconds": statistics.median(t["first_response_seconds"] for t in ts),
                   "last_response_seconds": statistics.median(t["last_response_seconds"] for t in ts),
                   "cpu_seconds": statistics.median(t["cpu_seconds"] for t in ts),
                   "cpu_mean_percent": statistics.median(t["cpu_mean_percent"] for t in ts),
                   "gpu_mean_percent": statistics.median(t["resources"]["gpu_mean_percent"] for t in ts),
                   "gpu_peak_percent": max(t["resources"]["gpu_peak_percent"] for t in ts),
                   "system_cpu_mean_percent": statistics.median(t["resources"]["system_cpu_mean_percent"] for t in ts),
                   "system_available_min_gib": min(t["resources"]["system_available_min_gib"] for t in ts),
                   "rss_peak_gib": max(t["resources"]["rss_peak_gib"] for t in ts),
                   "rss_mean_gib": statistics.median(t["resources"]["rss_mean_gib"] for t in ts),
                   "committed_peak_gib": max(t["resources"]["committed_peak_gib"] for t in ts),
                   "gpu_shared_peak_gib": max(t["resources"]["gpu_shared_peak_gib"] for t in ts),
                   "gpu_dedicated_peak_gib": max(t["resources"]["gpu_dedicated_peak_gib"] for t in ts),
                   "upload_bytes": ts[0]["upload_bytes"]}
            for phase in ("read_and_upload", "count_tokens", "predict", "remove_copy"):
                row[phase + "_seconds"] = statistics.median(sum(p["seconds"] for p in t["phases"] if p["kind"] == phase) for t in ts)
            rows.append(row)
        shared, separate = rows[-2:]
        paired = []
        for i in (1, 2, 3):
            pair = {t["mode"]: t for t in report["trials"] if t["case"] == case["id"] and t["round"] == i}
            a, b = pair["shared"]["wall_seconds"], pair["separate"]["wall_seconds"]
            paired.append({"round": i, "shared_seconds": a, "separate_seconds": b, "shared_minus_separate_seconds": a-b, "shared_change_percent": 100*(a/b-1)})
        comparisons.append({"case": case["id"], "median_time_change_percent": 100*(shared["wall_seconds"] / separate["wall_seconds"] - 1),
                            "paired_rounds": paired, "paired_change_median_percent": statistics.median(p["shared_change_percent"] for p in paired),
                            "upload_bytes_saved": separate["upload_bytes"] - shared["upload_bytes"]})
    return {"statistic": "Medians of 3 post-warmup workflows per case/mode; memory and GPU peaks are maxima across those 3 workflows", "rows": rows, "comparisons": comparisons}


def chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"shared": "#098477", "separate": "#3d65bd"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
    fields = [("wall_seconds", "Tiempo para las 3 respuestas", "segundos"),
              ("cpu_seconds", "Trabajo de CPU acumulado", "segundos de CPU"),
              ("committed_peak_gib", "Memoria privada comprometida: pico", "GiB"),
              ("gpu_mean_percent", "GPU: ocupación media del motor más activo", "%")]
    for ax, (field, title, unit) in zip(axes.flat, fields):
        for j, mode in enumerate(MODES):
            values = [next(r[field] for r in summary["rows"] if r["case"] == case and r["mode"] == mode) for case in LABELS]
            xs = [x + (j-.5)*.36 for x in range(3)]
            errors = None
            if field == "wall_seconds":
                rs = [next(r for r in summary["rows"] if r["case"] == case and r["mode"] == mode) for case in LABELS]
                errors = [[r[field] - r["min_seconds"] for r in rs], [r["max_seconds"] - r[field] for r in rs]]
            bars = ax.bar(xs, values, width=.34, color=colors[mode], label=MODES[mode], yerr=errors, capsize=3)
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_xticks(range(3), ["Imagen 45,9 MP", "Audio 30 s", "Vídeo 60 s"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("E2B QAT Q4 · 3 preguntas sobre el mismo archivo · sin caché de inferencia", fontsize=13)
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


def render(report, summary, verification):
    def f(value, digits=2):
        return f"{value:.{digits}f}".replace(".", ",")

    lines = ["# E2B: tres preguntas sobre el mismo archivo", "",
             "Comparación medida en el equipo local con Gemma E2B QAT Q4 y su componente multimodal. "
             "Se completaron 24 flujos y 72 inferencias. La primera ronda (18 inferencias) se excluye del resumen; quedan tres repeticiones por caso y modo.", "",
             "| Archivo | Modo | Total 3 respuestas (s) | Rango (s) | Primera (s) | CPU media (%) | CPU acumulada (s) | RAM RSS pico (GiB) | Compromiso privado pico (GiB) | GPU media (%) | RAM GPU compartida pico (GiB) |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["rows"]:
        lines.append(f"| {r['label']} | {MODES[r['mode']]} | {f(r['wall_seconds'])} | {f(r['min_seconds'])}–{f(r['max_seconds'])} | {f(r['first_response_seconds'])} | {f(r['cpu_mean_percent'], 1)} | {f(r['cpu_seconds'])} | {f(r['rss_peak_gib'])} | {f(r['committed_peak_gib'])} | {f(r['gpu_mean_percent'], 1)} | {f(r['gpu_shared_peak_gib'])} |")
    lines += ["", "Los tiempos y porcentajes son medianas de tres pasadas; los picos de memoria son el máximo observado entre esas pasadas.", "",
              "## Qué se comparó", "",
              "- Compartido: leer/subir una copia, comprobar los tokens de las tres preguntas, ejecutar tres inferencias secuenciales y retirar la copia.",
              "- Separado: leer/subir el mismo archivo, comprobar tokens, inferir y retirar la copia para cada pregunta.",
              "- Mismo archivo, mismas preguntas y opciones en ambos modos. Rotación de preguntas y alternancia del orden entre modos.",
              "- En ambos casos hay tres evaluaciones completas del modelo y del contenido multimodal. La interfaz actual no comparte los embeddings ni agrupa las tres preguntas en una sola inferencia.",
              "- Las 72 llamadas usaron cache_prompt=false y devolvieron cero tokens reutilizados. El modelo permaneció cargado y no se vació la caché de disco del sistema operativo.",
              "- Se incluye lectura local, transferencia HTTP, verificación de tokens, preprocesamiento, inferencia y limpieza. Se excluyen interacción humana y renderizado del navegador.", "",
              "## Desglose de tiempos", "", "| Archivo | Modo | Leer y subir (s) | Contar tokens externo (s) | 3 predict, con comprobación interna (s) | Retirar copias (s) |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["rows"]:
        lines.append(f"| {r['label']} | {MODES[r['mode']]} | {f(r['read_and_upload_seconds'])} | {f(r['count_tokens_seconds'])} | {f(r['predict_seconds'])} | {f(r['remove_copy_seconds'], 3)} |")
    lines += ["", "Cada columna es una mediana independiente; su suma puede diferir de la mediana del total.", "",
              "## Comparación emparejada", "",
              "Un cambio negativo significa menos tiempo al compartir el archivo. Con tres repeticiones, los rangos describen esta ejecución; no se afirma una mejora estadísticamente demostrada.", "",
              "| Archivo | Cambio de las medianas | Mediana del cambio emparejado | Cambios por ronda | Bytes transferidos, compartido / separado |", "|---|---:|---:|---|---|"]
    for c in summary["comparisons"]:
        rs = [r for r in summary["rows"] if r["case"] == c["case"]]
        deltas = "; ".join(f(p["shared_change_percent"], 1) + "%" for p in c["paired_rounds"])
        lines.append(f"| {LABELS[c['case']]} | {f(c['median_time_change_percent'], 1)}% | {f(c['paired_change_median_percent'], 1)}% | {deltas} | {rs[0]['upload_bytes']:,} / {rs[1]['upload_bytes']:,} |")
    lines += ["", "El cambio de las medianas y la mediana de los cambios emparejados son resúmenes distintos. Pueden diferir bastante, incluso de signo, cuando las rondas tienen velocidades diferentes; por eso se conservan los tres pares y los rangos, sin afirmar una aceleración estable.", "",
              "Compartir una copia reduce la transferencia del archivo en un 66,7 %. La comprobación de todas las preguntas antes de inferir puede retrasar la primera respuesta.", "",
              "## Cómo interpretar los recursos", "",
              "- CPU: tiempo de usuario y kernel del servicio y sus hijos (Job Object de Windows, incluidos ffmpeg/ffprobe ya terminados), más backend y cliente de medición. El porcentaje se normaliza sobre los 16 procesadores lógicos: 100 % significa ocupar toda esa capacidad.",
              "- RAM RSS: suma de memoria residente de esos procesos, incluido el monitor. El compromiso privado mide memoria privada respaldada por RAM o archivo de paginación; no equivale a páginas residentes. Los valores RSS de 1,08–1,92 GiB de esta sesión no significan que el modelo solo necesite esa cantidad: el compromiso privado llegó a 6,25–7,11 GiB y la GPU mantuvo RAM compartida. Se informa memoria total de los procesos, no un incremento sobre reposo.",
              "- GPU: contadores PDH por PID de la Intel Arc 140T. Se informa la ocupación del motor más activo, ponderada por tiempo; no una suma de motores ni el consumo eléctrico.",
              "- La GPU es integrada: su RAM compartida se solapa con la RAM del equipo. No se debe sumar a RSS como si fuese memoria adicional independiente.",
              "- E4B permaneció cargado e inactivo. Sus procesos están excluidos de las cifras atribuidas a E2B; otras aplicaciones pueden influir en los tiempos.",
              f"- El equipo tenía otras cargas: las medianas de CPU global por escenario estuvieron entre {f(min(r['system_cpu_mean_percent'] for r in summary['rows']), 1)} y {f(max(r['system_cpu_mean_percent'] for r in summary['rows']), 1)} %. Estos resultados describen la sesión actual, no un equipo aislado de otras tareas.",
              f"- Muestreo solicitado: 0,1 s. Intervalo real mediano {f(report['sample_intervals']['median_seconds'], 3)} s; máximo {f(report['sample_intervals']['max_seconds'], 3)} s. Las ráfagas por debajo de ese intervalo pueden no quedar reflejadas en los picos.",
              "- Son preguntas de selección de opción con una etiqueta de salida y probabilidades independientes. No mide generación de respuestas largas ni precisión/calibración.", "",
              "## Archivos y trazabilidad", "",
              "Fotografía de los Alpes de 10.109 × 4.542 px (14.679.474 bytes); voz humana en español de 30 s (1.323.078 bytes); vídeo real de Marsella de 60 s (20.289.393 bytes), muestreado a 16 fotogramas. El adaptador usa audio mono de 16 kHz y no analiza el sonido del vídeo.", "",
              "Las preguntas, opciones, rutas, hashes de medios, configuración y fuentes congeladas están en plan.json. results.json contiene cada resultado y los intervalos; samples.jsonl contiene las lecturas originales; verification.json comprueba los datos de forma independiente. No se cambió el adaptador ni el modelo para esta prueba.", "",
              f"Verificación: {'CORRECTA' if verification['all_passed'] else 'FALLIDA'}, {len(verification['checks'])} comprobaciones. Las 48 copias temporales de esta prueba fueron retiradas.", "",
              "Fuentes y licencias: ../real-media-benchmark-20261017/ATTRIBUTION.md.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf8")
    image = base64.b64encode((OUT / "comparison.png").read_bytes()).decode("ascii")
    # A self-contained local HTML file; no network, scripts, or changes to the running UI.
    body = []
    in_table = False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if not in_table:
                body.append('<div class="table"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in cells) + '</tr></thead><tbody>')
                in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
            continue
        if in_table:
            body.append("</tbody></table></div>")
            in_table = False
        if line.startswith("# "):
            body.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line.startswith("## "):
            body.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("- "):
            body.append('<p class="point">• ' + html.escape(line[2:]) + "</p>")
        elif line:
            body.append("<p>" + html.escape(line) + "</p>")
    body.insert(2, f'<img alt="Comparación de tiempo, CPU, RAM y GPU" src="data:image/png;base64,{image}">')
    links = ' · '.join(f'<a href="{name}">{html.escape(label)}</a>' for name, label in (("summary.json", "Resumen JSON"), ("trials.csv", "24 flujos CSV"), ("verification.json", "Verificación"), ("results.json", "Resultados completos"), ("samples.jsonl", "Muestreo original"), ("../real-media-benchmark-20261017/ATTRIBUTION.md", "Fuentes y licencias")))
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>E2B: varias preguntas y consumo</title><style>body{font:16px/1.55 system-ui;color:#19302e;background:#f4f7f6;max-width:1280px;margin:2rem auto;padding:0 1.5rem}h1,h2{line-height:1.2}h2{margin-top:2.2rem}img{width:100%;height:auto;background:white;border-radius:10px}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:.75rem;border-bottom:1px solid #d7e2de}th{background:#e3efeb}.point{margin:.5rem 0}a{color:#076c63}nav{padding:1rem 0}</style><nav>' + links + '</nav>' + '\n'.join(body) + '</html>'
    (OUT / "report.html").write_text(doc, encoding="utf8")


def main():
    report = json.loads((OUT / "results.json").read_text(encoding="utf8"))
    samples = [json.loads(line) for line in (OUT / "samples.jsonl").read_text(encoding="utf8").splitlines()]
    verification = verify(report, samples)
    save("verification.json", verification)
    assert verification["all_passed"], verification["checks"]
    summary = summarize(report)
    summary["verification"] = "verification.json"
    summary["idle_resources"] = report["idle_resources"]
    summary["sample_intervals"] = report["sample_intervals"]
    save("summary.json", summary)
    with (OUT / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "mode", "round", "warmup", "wall_seconds", "first_response_seconds", "last_response_seconds", "cpu_seconds", "cpu_mean_percent", "rss_peak_gib", "committed_peak_gib", "gpu_shared_peak_gib", "gpu_mean_percent", "gpu_peak_percent", "upload_bytes"))
        writer.writeheader()
        for t in report["trials"]:
            writer.writerow({key: t["round"] == 0 if key == "warmup" else t[key] if key in t else t["resources"][key] for key in writer.fieldnames})
    chart(summary)
    render(report, summary, verification)
    print(json.dumps({"verification": verification, "summary": summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
