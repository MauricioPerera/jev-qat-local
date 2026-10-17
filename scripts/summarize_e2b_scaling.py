"""Independent verification and reports for the 5/10/20-question resource test."""
import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from summarize_e2b_shared_file import resources
from jev_qat.config import save_json

LABELS = {"image_alps": "Imagen 45,9 MP", "audio_30": "Audio 30 s", "video_60_16": "Vídeo 60 s / 16 fotogramas"}
MODES = {"baseline": "Sin reutilización", "optimized": "Optimizado"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right, category, case, n):
    a, b = left["result"], right["result"]
    assert set(a["probabilities"]) == set(b["probabilities"])
    return {"comparison": category, "case": case, "count": n, "question_id": left["question_id"],
            "same_tokens": left["budget"]["input_tokens"] == right["budget"]["input_tokens"],
            "same_prediction": a["prediction"] == b["prediction"],
            "max_absolute_probability_delta": max(abs(p-b["probabilities"][k]) for k, p in a["probabilities"].items())}


def verify(out, report, samples):
    ts = report["trials"]
    main = [t for t in ts if t["purpose"] == "main"]
    bank = {c["id"]: c for c in report["cases"]}
    checks = {
        "measurement_checks": all(report["checks"].values()),
        "18_unique_main_workflows": len(main) == len({(t["case"], t["count"], t["mode"]) for t in main}) == 18,
        "279_completed_predictions_210_main": sum(len(t["answers"]) for t in ts) == 279 and sum(len(t["answers"]) for t in main) == 210,
        "20_distinct_questions_per_file": all(len(c["questions"]) == len({q["question"] for q in c["questions"]}) == 20 for c in bank.values()),
        "all_measured_sources_unchanged": all(sha(ROOT/p) == digest == sha(out/"source"/p) for p, digest in report["frozen_sources"].items()),
        "measurement_code_matches_archive": all(sha(ROOT/p) == sha(out/"source"/p) for p in ("scripts/benchmark_e2b_scaling.py", "scripts/e2b_scaling_questions.py", "scripts/benchmark_e2b_shared_file.py", "scripts/benchmark_multimodal_resources.py")),
        "same_real_media": all(sha(c["source"]) == c["sha256"] for c in bank.values()),
        "contiguous_raw_samples": all(a["end"] == b["start"] for a, b in zip(samples, samples[1:])),
        "valid_sampling_intervals": bool(samples) and all(s["end"] > s["start"] and 0 <= s["gpu_busy_percent"] <= 100 for s in samples),
        "workflows_do_not_overlap": all(a["end"] <= b["start"] for a, b in zip(ts, ts[1:])),
        "cpu_job_snapshot_formula": all(math.isclose(t["cpu_seconds"], t["cpu_after"]["seconds"]-t["cpu_before"]["seconds"], abs_tol=1e-9)
            and math.isclose(t["cpu_mean_percent"], 100*t["cpu_seconds"]/t["wall_seconds"]/report["logical_processors"], abs_tol=1e-9) for t in ts),
    }
    intervals_ok = True
    full_counts = vectors = cache = prep = phases = True
    tokens = []
    for t in ts:
        n, optimized = t["count"], t["mode"] == "optimized"
        phases &= len(t["phases"]) == 2*n+2 and [p["kind"] for p in t["phases"]] == ["read_and_upload", *(["count_tokens"]*n), *(["predict"]*n), "remove_copy"]
        phases &= len(t["order"]) == len(set(t["order"])) == n and sorted(t["order"]) == list(range(n))
        phases &= [a["question_id"] for a in t["answers"]] == t["order"]
        for interval in (t, *t["phases"]):
            independent = resources(samples, interval["start"], interval["end"])
            intervals_ok &= abs(independent["covered_seconds"] - (interval["end"]-interval["start"])) < 1e-6
            intervals_ok &= all(math.isclose(value, interval["resources"][k], abs_tol=1e-8) for k, value in independent.items())
        for i, a in enumerate(t["answers"]):
            b, result = a["budget"], a["result"]
            m, probs = result["metadata"], result["probabilities"]
            usage = m["usage"]
            options = bank[t["case"]]["questions"][a["question_id"]]["options"]
            vectors &= set(probs) == set(options) and all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values()) and abs(sum(probs.values())-1) < 1e-6
            vectors &= result["prediction"] == options[result["prediction_index"]] and math.isclose(result["confidence"], max(probs.values()), abs_tol=1e-10)
            full_counts &= b["input_tokens"] == usage["prompt_tokens"] == m["context_budget"]["input_tokens"]
            full_counts &= b["total_tokens"] == m["context_budget"]["total_tokens"] == b["input_tokens"]+1 <= 4096
            full_counts &= usage["completion_tokens"] == b["reserved_output_tokens"] == 1 and b["context_limit_tokens"] == 4096
            full_counts &= b["inference_executed"] is False and b["available_tokens"] == 4096-b["total_tokens"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            cache &= cached == m["timings"]["cache_n"] and (cached > 0) is (optimized and i > 0)
            cache &= cached + m["timings"]["prompt_n"] == usage["prompt_tokens"]
            prep &= b["media_preparation"]["hit"] is (optimized and i > 0)
            prep &= m["media_preparation"]["hit"] is optimized and m["media_preparation"]["enabled"] is optimized
            vectors &= m["backend"] == "gemma-4-E2B-it-qat-q4_0-gguf" and m["calibrated"] is False
            tokens.append(b["input_tokens"])
    checks.update({"all_intervals_recalculated_from_raw_samples": intervals_ok,
                   "all_279_finite_independent_option_vectors": vectors, "all_279_exact_full_context_checks": full_counts,
                   "cache_reset_then_real_prefix_reuse": cache, "single_preparation_per_optimized_workflow": prep,
                   "exact_own_question_order_and_workflow_phases": phases})
    comparisons = []
    codec = []
    for case in bank:
        optimized20 = next(t for t in main if t["case"] == case and t["count"] == 20 and t["mode"] == "optimized")
        ref20 = {a["question_id"]: a for a in optimized20["answers"]}
        baseline20 = next(t for t in main if t["case"] == case and t["count"] == 20 and t["mode"] == "baseline")
        alone20 = {a["question_id"]: a for a in baseline20["answers"]}
        for n in (5, 10, 20):
            pair = {t["mode"]: t for t in main if t["case"] == case and t["count"] == n}
            refs = {a["question_id"]: a for a in pair["baseline"]["answers"]}
            for a in pair["optimized"]["answers"]:
                comparisons.append(compare(a, refs[a["question_id"]], "paired_mode", case, n))
                if n < 20:
                    comparisons.append(compare(a, ref20[a["question_id"]], "cross_size", case, n))
            if case != "image_alps":
                counts = {m: t["cpu_after"]["job_processes"]-t["cpu_before"]["job_processes"] for m, t in pair.items()}
                codec.append(counts["optimized"] > 0 and counts["baseline"] == 2*n*counts["optimized"])
        reverse = next(t for t in ts if t["case"] == case and t["purpose"] == "reverse_audit")
        assert reverse["order"] == list(reversed(range(20)))
        for a in reverse["answers"]:
            comparisons.append(compare(a, alone20[a["question_id"]], "reverse_vs_uncached", case, 20))
    checks["codec_processes_match_2N_to_one_preparation_reduction"] = all(codec)
    checks["105_mode_pairs_45_cross_size_60_reverse_comparisons"] = all(sum(c["comparison"] == kind for c in comparisons) == n for kind, n in (("paired_mode", 105), ("cross_size", 45), ("reverse_vs_uncached", 60)))
    checks["all_comparison_token_counts_identical"] = all(c["same_tokens"] for c in comparisons)
    checks["all_comparison_top_choices_identical"] = all(c["same_prediction"] for c in comparisons)
    maximum_delta = max(c["max_absolute_probability_delta"] for c in comparisons)
    checks["all_probability_deltas_below_one_percentage_point"] = maximum_delta < .01
    checks["all_10_and_20_question_groups_exceed_4096_in_sum_but_fit_individually"] = all(sum(a["budget"]["input_tokens"] for a in t["answers"]) > 4096 and all(a["budget"]["total_tokens"] <= 4096 for a in t["answers"]) for t in main if t["count"] >= 10)
    controls = json.loads((out / "budget-controls.json").read_text(encoding="utf8"))
    checks["live_boundary_overflow_and_continuation_controls"] = controls["passed"] and all(controls["checks"].values())
    uploads = ROOT / "integration/jev-qat-e2b/uploads"
    ids = [t["upload_id"] for t in ts] + [c["upload_id"] for c in controls["cases"]]
    checks["all_27_owned_uploads_removed"] = len(ids) == len(set(ids)) == 27 and all(not list(uploads.glob(i+".*")) for i in ids)
    state = json.loads((ROOT / "integration/jev-qat-e2b/service.json").read_text(encoding="utf8"))
    checks["same_service_and_backend"] = state["pid"] == report["service_pid"] and state["backend_pid"] == report["backend_pid"]
    with httpx.Client(timeout=10, trust_env=False) as client:
        checks["e2b_service_healthy"] = client.get(report["config"]["api_url"] + "/health").status_code == 200
    return {"all_passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "max_absolute_probability_delta": maximum_delta,
            "main_input_token_range": [min(tokens), max(tokens)], "raw_sample_count": len(samples)}


def summarize(report):
    rows = []
    for t in report["trials"]:
        if t["purpose"] != "main":
            continue
        r = {k: t[k] for k in ("case", "mode", "count", "wall_seconds", "first_response_seconds", "last_response_seconds", "cpu_seconds", "cpu_mean_percent")}
        for key in ("rss_peak_gib", "rss_mean_gib", "committed_peak_gib", "gpu_shared_peak_gib", "gpu_mean_percent", "gpu_peak_percent", "system_cpu_mean_percent"):
            r[key] = t["resources"][key]
        r["per_question_seconds"] = t["wall_seconds"]/t["count"]
        r["questions_per_minute"] = 60*t["count"]/t["wall_seconds"]
        r["input_tokens_min"] = min(a["budget"]["input_tokens"] for a in t["answers"])
        r["input_tokens_max"] = max(a["budget"]["input_tokens"] for a in t["answers"])
        r["input_tokens_sum"] = sum(a["budget"]["input_tokens"] for a in t["answers"])
        r["cached_tokens_sum"] = sum(a["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] for a in t["answers"])
        r["predict_median_seconds"] = statistics.median(p["seconds"] for p in t["phases"] if p["kind"] == "predict")
        later = [p["seconds"] for p in t["phases"] if p["kind"] == "predict"][1:]
        r["later_question_median_seconds"] = statistics.median(later)
        for kind in ("read_and_upload", "count_tokens", "predict", "remove_copy"):
            r[kind+"_seconds"] = sum(p["seconds"] for p in t["phases"] if p["kind"] == kind)
        rows.append(r)
    rows.sort(key=lambda r: (list(LABELS).index(r["case"]), r["count"], r["mode"]))
    comparisons = []
    for case in LABELS:
        for n in (5, 10, 20):
            pair = {r["mode"]: r for r in rows if r["case"] == case and r["count"] == n}
            a, b = pair["baseline"], pair["optimized"]
            comparisons.append({"case": case, "count": n,
                                "time_saved_percent": 100*(1-b["wall_seconds"]/a["wall_seconds"]),
                                "cpu_seconds_saved_percent": 100*(1-b["cpu_seconds"]/a["cpu_seconds"]),
                                "rss_peak_change_mib": (b["rss_peak_gib"]-a["rss_peak_gib"])*1024,
                                "commit_peak_change_mib": (b["committed_peak_gib"]-a["committed_peak_gib"])*1024})
    return {"statistic": "One measured paired workflow per case and size, not a median or extrapolation; reverse-order repeats are reported separately.", "rows": rows, "comparisons": comparisons}


def charts(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    fields = [("wall_seconds", "Tiempo total", "s"), ("cpu_seconds", "CPU acumulada", "s de CPU"),
              ("committed_peak_gib", "Memoria privada comprometida: pico", "GiB"), ("gpu_mean_percent", "GPU: ocupación media del motor más activo", "%")]
    for col, case in enumerate(LABELS):
        for row, (key, title, unit) in enumerate(fields):
            ax = axes[row, col]
            for mode, color in (("baseline", "#3d65bd"), ("optimized", "#098477")):
                points = sorted((r for r in summary["rows"] if r["case"] == case and r["mode"] == mode), key=lambda r: r["count"])
                ax.plot([r["count"] for r in points], [r[key] for r in points], "o-", color=color, label=MODES[mode])
            ax.set_title((LABELS[case]+"\n" if row == 0 else "")+title, fontsize=10)
            ax.set_xticks([5, 10, 20])
            ax.set_ylabel(unit)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 3:
                ax.set_xlabel("Preguntas sobre el mismo archivo")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("E2B QAT Q4 · escalado real de 5, 10 y 20 preguntas independientes")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for ax, case in zip(axes, LABELS):
        for mode, color in (("baseline", "#3d65bd"), ("optimized", "#098477")):
            points = sorted((r for r in summary["rows"] if r["case"] == case and r["mode"] == mode), key=lambda r: r["count"])
            ax.plot([r["count"] for r in points], [r["first_response_seconds"] for r in points], "o-", label=MODES[mode], color=color)
        ax.set(title=LABELS[case], xlabel="Preguntas", ylabel="s hasta la primera respuesta")
        ax.set_xticks([5, 10, 20])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Se comprueban los tokens de todas las preguntas antes de la primera inferencia")
    fig.savefig(out / "first-response.png", dpi=160)
    plt.close(fig)


def render(out, report, summary, verified):
    f = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")
    lines = ["# E2B: 5, 10 y 20 preguntas sobre el mismo archivo", "",
             "Prueba real de la versión optimizada frente al mismo adaptador con reutilización desactivada. Una pasada completa por cada archivo, tamaño y modo; los valores no son extrapolaciones ni medianas. Se añadieron calentamientos separados y una repetición de 20 preguntas en orden inverso por archivo.", "",
             "Se completaron 279 inferencias: 210 en la matriz principal, 60 para comprobar el orden inverso y 9 de calentamiento. Las pruebas adicionales de límite de contexto se hicieron después y no forman parte de los tiempos de rendimiento.", "",
             "## Tiempo total y recursos", "", "Cada celda muestra sin reutilización → optimizado. El tiempo incluye una carga del archivo, todos los conteos externos, inferencias secuenciales con su propio conteo interno y retirada de la copia.", "",
             "| Archivo | Preguntas | Tiempo total (s) | CPU media (%) | CPU acumulada (s) | RAM residente pico (GiB) | Compromiso privado pico (GiB) | GPU media (%) | RAM compartida GPU pico (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in LABELS:
        for n in (5, 10, 20):
            pair = {r["mode"]: r for r in summary["rows"] if r["case"] == case and r["count"] == n}
            a, b = pair["baseline"], pair["optimized"]
            cells = [LABELS[case], str(n)]
            for key in ("wall_seconds", "cpu_mean_percent", "cpu_seconds", "rss_peak_gib", "committed_peak_gib", "gpu_mean_percent", "gpu_shared_peak_gib"):
                digits = 1 if key.endswith("percent") else 2
                cells.append(f(a[key], digits)+" → "+f(b[key], digits))
            lines.append("| "+" | ".join(cells)+" |")
    lines += ["", "## Detalle del modo optimizado", "", "| Archivo | Preguntas | Primera respuesta (s) | Total por pregunta (s) | Inferencias posteriores, mediana (s) | Preguntas/minuto | Entrada por pregunta (tokens) | Suma de entradas (tokens) | Ahorro de tiempo (%) |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in summary["rows"]:
        if r["mode"] != "optimized":
            continue
        comparison = next(c for c in summary["comparisons"] if c["case"] == r["case"] and c["count"] == r["count"])
        cells = [LABELS[r["case"]], str(r["count"]), f(r["first_response_seconds"]), f(r["per_question_seconds"]), f(r["later_question_median_seconds"]), f(r["questions_per_minute"], 1), f"{r['input_tokens_min']}–{r['input_tokens_max']}", str(r["input_tokens_sum"]), f(comparison["time_saved_percent"], 1)]
        lines.append("| "+" | ".join(cells)+" |")
    lines += ["", "La media total por pregunta incluye carga, conteos y primera inferencia. La mediana de inferencias posteriores excluye esos pasos iniciales y describe las preguntas 2…N. La primera respuesta puede tardar más al aumentar N, porque el flujo comprueba todos los presupuestos antes de empezar a inferir.", "",
              "## Probabilidades independientes", "", f"Verificación: {'CORRECTA' if verified['all_passed'] else 'FALLIDA'}, {len(verified['checks'])} comprobaciones. Se contrastaron 105 parejas optimizado/sin reutilización, 45 comparaciones entre tamaños y 60 preguntas en orden inverso contra su control sin caché. Máxima diferencia absoluta entre probabilidades: {f(100*verified['max_absolute_probability_delta'], 6)} puntos porcentuales.",
              "Cada pregunta tiene su propia consulta, entre dos y cuatro opciones en esta prueba, vector finito normalizado y salida de una etiqueta. Se variaron las posiciones de las opciones. El banco contiene 20 preguntas distintas por archivo, incluidas preguntas relacionadas entre sí. No se incluyeron otras preguntas ni respuestas en el prompt de cada petición. Las probabilidades permanecen sin calibrar; esta prueba no mide precisión semántica con etiquetas humanas.", "",
              "## Límite de 4096 tokens", "",
              "Cada respuesta se reservó un token. Todos los conteos externos coincidieron con el consumo nativo y con el conteo interno previo a inferir. La reutilización no descuenta tokens del presupuesto: se cuentan también los tokens del prefijo reutilizado.",
              "En los tres tipos de archivo se buscó y ejecutó una petición de exactamente 4095 tokens de entrada + 1 de salida, aceptada con total 4096. Una entrada de 4096 + 1 de salida se rechazó con HTTP 422; el registro del motor confirmó que no arrancó inferencia para ella. Después de cada rechazo se ejecutó una pregunta válida correspondiente a los grupos de 5, 10 y 20. Los nueve controles se conservaron en budget-controls.json.",
              "La suma de los presupuestos de preguntas distintas no constituye un único contexto. Los grupos de 10 y 20 superan 4096 tokens sumados y funcionan porque cada pregunta cabe individualmente.", "",
              "## Protocolo y alcance", "",
              "- Mismos archivos públicos: foto de los Alpes de 10.109 × 4.542 píxeles, voz en español de 30 segundos y vídeo de Marsella de 60 segundos a 16 fotogramas; el vídeo se evalúa sin su pista de audio. Procedencia y licencias: ../real-media-benchmark-20261017/ATTRIBUTION.md.",
              "- Ambas variantes suben una copia por grupo. El control usa reuse_media=false y cache_prompt=false siempre. El optimizado usa reuse_media=true, empieza con cache_prompt=false y habilita la reutilización nativa a partir de la segunda pregunta. Cada grupo empieza sin ventaja de la inferencia anterior.",
              "- Orden alternado entre modos; tamaños distribuidos entre modalidades para reducir la influencia de la deriva temporal. La repetición en orden inverso se informa como auditoría de independencia, no se mezcla con la pasada principal.",
              "- Modelo ya cargado. No se vació la caché de disco del sistema operativo. No se cambió el adaptador, el modelo ni su configuración durante la prueba.",
              "- CPU: usuario + kernel del servicio y sus descendientes, incluso procesos de conversión terminados, más backend y cliente/monitor. Porcentaje normalizado sobre 16 procesadores lógicos. Los segundos de CPU expresan trabajo acumulado; su porcentaje puede subir durante una ejecución más breve.",
              "- RAM residente es la suma de working sets. El compromiso privado puede estar respaldado por RAM o paginación. No se trata de memoria incremental respecto al reposo. La GPU integrada comparte RAM: no sumar sus GiB como memoria independiente.",
              "- GPU: contador PDH por PID del motor más activo, ponderado por tiempo, y pico de memoria compartida. No mide vatios ni energía. E4B permaneció cargado e inactivo; sus procesos se excluyen del consumo atribuido.",
              f"- Muestreo solicitado cada 0,1 s; intervalo real mediano {f(report['sample_intervals']['median_seconds'], 3)} s, máximo {f(report['sample_intervals']['max_seconds'], 3)} s. Otras aplicaciones siguieron activas.",
              "- Una pasada emparejada por tamaño: estos valores describen esta sesión y no establecen significación estadística. Cada punto de la curva se midió por completo.", "",
              "plan.json conserva preguntas, opciones, hashes y protocolo; source/ archiva las fuentes medidas; results.json incluye cada respuesta; samples.jsonl, el muestreo; verification.json, las comprobaciones; workflows.csv y questions.csv permiten analizar los datos.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf8")
    body, table = [], False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            tag = "td" if table else "th"
            if not table:
                body.append('<div class="scroll"><table>')
                table = True
            body.append("<tr>"+"".join(f"<{tag}>{c}</{tag}>" for c in cells)+"</tr>")
            continue
        if table:
            body.append("</table></div>")
            table = False
        if line.startswith("# "):
            body.append("<h1>"+html.escape(line[2:])+"</h1>")
        elif line.startswith("## "):
            body.append("<h2>"+html.escape(line[3:])+"</h2>")
        elif line:
            body.append("<p>"+html.escape(line.removeprefix("- "))+"</p>")
    for index, name, alt in ((2, "comparison.png", "Escalado de tiempo, CPU, RAM y GPU"), (4, "first-response.png", "Tiempo hasta la primera respuesta")):
        data = base64.b64encode((out/name).read_bytes()).decode()
        body.insert(index, f'<img alt="{alt}" src="data:image/png;base64,{data}">')
    nav = ' · '.join(f'<a href="{p}">{label}</a>' for p, label in (("summary.json", "Resumen"), ("workflows.csv", "CSV por grupo"), ("questions.csv", "CSV por pregunta"), ("verification.json", "Verificación"), ("budget-controls.json", "Control de tokens"), ("../real-media-benchmark-20261017/ATTRIBUTION.md", "Licencias")))
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>E2B: 5, 10 y 20 preguntas</title><style>body{font:16px/1.55 system-ui;max-width:1450px;margin:2rem auto;padding:0 1.5rem;background:#f4f7f6;color:#19302e}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:14px}th,td{padding:.7rem;text-align:left;border-bottom:1px solid #d7e2de}th{background:#e3efeb}h2{margin-top:2rem}a{color:#076c63}</style><nav>'+nav+'</nav>'+'\n'.join(body)+'</html>'
    (out / "report.html").write_text(doc, encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "inference/e2b-scaling-5-10-20")
    out = parser.parse_args().output.resolve()
    report = json.loads((out / "results.json").read_text(encoding="utf8"))
    samples = [json.loads(s) for s in (out / "samples.jsonl").read_text(encoding="utf8").splitlines()]
    verified = verify(out, report, samples)
    save_json(out / "verification.json", verified)
    assert verified["all_passed"], verified["checks"]
    summary = summarize(report)
    save_json(out / "summary.json", summary)
    with (out / "workflows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    with (out / "questions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "count", "mode", "purpose", "position", "question_id", "response_at", "predict_seconds", "input_tokens", "cached_tokens", "prediction", "probabilities_json"))
        writer.writeheader()
        for t in report["trials"]:
            for i, a in enumerate(t["answers"]):
                phase = next(p for p in t["phases"] if p["kind"] == "predict" and p["question"] == a["question_id"])
                writer.writerow({"case": t["case"], "count": t["count"], "mode": t["mode"], "purpose": t["purpose"], "position": i+1,
                                 "question_id": a["question_id"], "response_at": a["response_at"], "predict_seconds": phase["seconds"],
                                 "input_tokens": a["budget"]["input_tokens"], "cached_tokens": a["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"],
                                 "prediction": a["result"]["prediction"], "probabilities_json": json.dumps(a["result"]["probabilities"], ensure_ascii=False)})
    charts(out, summary)
    render(out, report, summary, verified)
    print(json.dumps({"verified": True, "checks": len(verified["checks"]), "max_probability_delta": verified["max_absolute_probability_delta"], "summary": summary}, ensure_ascii=True))


if __name__ == "__main__":
    main()
