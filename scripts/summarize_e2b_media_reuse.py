"""Verify paired raw resource measurements and publish the optimization comparison."""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import httpx

from summarize_e2b_shared_file import resources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference/e2b-prepare-reuse/benchmark"
LABELS = {"image_alps": "Imagen 45,9 MP", "audio_30": "Audio 30 s", "video_60_16": "Vídeo 60 s / 16 fotogramas"}
MODES = {"baseline": "Anterior", "optimized": "Optimizado"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def verify(report, samples):
    trials = report["trials"]
    checks = {"measurement_checks": all(report["checks"].values()),
              "24_unique_complete_workflows": len(trials) == len({(t["case"], t["mode"], t["round"]) for t in trials}) == 24 and all(t["error"] is None and len(t["answers"]) == 3 for t in trials),
              "identical_source_hashes": all(sha(ROOT/p) == d for p, d in report["frozen_sources"].items()),
              "real_files_unchanged": all(sha(c["source"]) == c["sha256"] for c in report["cases"]),
              "raw_samples_contiguous": all(a["end"] == b["start"] for a, b in zip(samples, samples[1:])),
              "workflows_do_not_overlap": all(a["end"] <= b["start"] for a, b in zip(trials, trials[1:]))}
    old_plan = json.loads((ROOT / "inference/e2b-shared-file-benchmark/plan.json").read_text(encoding="utf8"))
    checks["baseline_snapshot_matches_preoptimization_version"] = all(
        sha(ROOT / "inference/e2b-prepare-reuse/baseline" / p) == old_plan["frozen_sources"][str(ROOT / p)]
        for p in ("jev_qat/e2b.py", "jev_qat/e2b_budget.py", "jev_qat/e2b_web.py", "jev_qat/web/e2b.js", "jev_qat_e2b_config.json"))
    checks["e4b_model_and_classifier_unchanged"] = all(
        sha(ROOT/p) == old_plan["frozen_sources"][str(ROOT/p)] for p in ("jev_qat/adapter.py", "jev_qat_config.json", "classification/current.json"))
    checks["cpu_job_accounting_recalculated"] = all(math.isclose(t["cpu_seconds"], t["cpu_after"]["seconds"]-t["cpu_before"]["seconds"], abs_tol=1e-9)
        and math.isclose(t["cpu_mean_percent"], 100*t["cpu_seconds"]/t["wall_seconds"]/report["logical_processors"], abs_tol=1e-9) for t in trials)
    codec_pairs = []
    for case in ("audio_30", "video_60_16"):
        for index in range(4):
            ts = {t["mode"]: t for t in trials if t["case"] == case and t["round"] == index}
            counts = {m: t["cpu_after"]["job_processes"]-t["cpu_before"]["job_processes"] for m, t in ts.items()}
            codec_pairs.append(counts["optimized"] > 0 and counts["baseline"] == 6*counts["optimized"])
    checks["codec_process_creations_reduced_sixfold"] = all(codec_pairs) and all(
        t["cpu_after"]["job_processes"] == t["cpu_before"]["job_processes"] for t in trials if t["case"] == "image_alps")
    proof = json.loads((OUT.parent / "baseline-payload-equivalence.json").read_text(encoding="utf8"))
    checks["disabled_optimization_emits_archived_native_requests"] = proof["passed"] and proof["all_native_post_payloads_identical"]
    trace_ok = True
    for t in trials:
        for interval in (t, *t["phases"]):
            independent = resources(samples, interval["start"], interval["end"])
            trace_ok &= abs(independent["covered_seconds"]-(interval["end"]-interval["start"])) < 1e-6
            trace_ok &= all(math.isclose(value, interval["resources"][key], abs_tol=1e-8) for key, value in independent.items())
    checks["all_resource_intervals_recomputed"] = trace_ok
    counts_ok = vectors_ok = cache_ok = preparation_ok = True
    for t in trials:
        optimized = t["mode"] == "optimized"
        for i, a in enumerate(t["answers"]):
            budget, result = a["budget"], a["result"]
            m = result["metadata"]
            counts_ok &= m["usage"]["prompt_tokens"] == budget["input_tokens"] and m["usage"]["completion_tokens"] == 1 and m["context_budget"]["total_tokens"] == budget["total_tokens"] <= 4096
            counts_ok &= a["budget"]["inference_executed"] is False
            opts = next(c for c in report["cases"] if c["id"] == t["case"])["questions"][a["question_id"]]["options"]
            p = result["probabilities"]
            vectors_ok &= set(p) == set(opts) and all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()) and abs(sum(p.values())-1) < 1e-6
            cached = m["usage"]["prompt_tokens_details"]["cached_tokens"]
            cache_ok &= cached == m["timings"]["cache_n"] and (cached > 0) is (optimized and i > 0)
            if optimized and i > 0:
                cache_ok &= cached > budget["input_tokens"] - 150  # Includes the media, not just a few template tokens.
            preparation_ok &= m["media_preparation"]["hit"] is optimized and m["media_preparation"]["enabled"] is optimized
            preparation_ok &= t["budgets"][i]["media_preparation"]["hit"] is (optimized and i > 0)
        preparation_ok &= len(t["phases"]) == 8 and [p["kind"] for p in t["phases"]] == ["read_and_upload", *(["count_tokens"]*3), *(["predict"]*3), "remove_copy"]
    checks.update({"each_question_has_exact_full_token_check": counts_ok, "independent_finite_option_distributions": vectors_ok,
                   "first_inference_cold_then_real_multimodal_prefix_reuse": cache_ok,
                   "one_preparation_in_optimized_flow_vs_six_in_baseline": preparation_ok})
    uploads = ROOT / "integration/jev-qat-e2b/uploads"
    ids = [t["upload_id"] for t in trials]
    checks["24_unique_uploads_removed"] = len(set(ids)) == 24 and all(not list(uploads.glob(identifier + ".*")) for identifier in ids)
    paired = []
    for case in LABELS:
        for r in range(4):
            modes = {t["mode"]: t for t in trials if t["case"] == case and t["round"] == r}
            baseline = {a["question_id"]: a for a in modes["baseline"]["answers"]}
            for a in modes["optimized"]["answers"]:
                b = baseline[a["question_id"]]
                paired.append({"case": case, "round": r, "question_id": a["question_id"],
                               "same_input_tokens": a["budget"]["input_tokens"] == b["budget"]["input_tokens"],
                               "same_prediction": a["result"]["prediction"] == b["result"]["prediction"],
                               "max_absolute_probability_delta": max(abs(v - b["result"]["probabilities"][k]) for k, v in a["result"]["probabilities"].items())})
    checks["all_36_paired_question_budgets_match"] = all(p["same_input_tokens"] for p in paired)
    checks["all_36_paired_predictions_match"] = all(p["same_prediction"] for p in paired)
    max_delta = max(p["max_absolute_probability_delta"] for p in paired)
    checks["probability_deltas_below_001_absolute"] = max_delta < .01
    live = json.loads((OUT.parent / "native-budget-check.json").read_text(encoding="utf8"))
    checks["prepared_media_overflow_rejected_before_native_inference"] = (
        live["passed"] and live["predict_status"] == 422 and live["native_logging_positive_control"]
        and live["native_inference_slots_started_for_overflow"] == 0
        and live["oversized_budget_using_same_prepared_file"]["media_preparation"]["hit"])
    suites = ET.parse(OUT.parent / "unit-tests.xml").getroot().findall(".//testsuite")
    unit_count = sum(int(s.attrib["tests"]) for s in suites)
    checks["all_167_python_regression_tests_passed"] = unit_count == 167 and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    with httpx.Client(timeout=10, trust_env=False) as client:
        checks["services_healthy"] = all(client.get(f"http://127.0.0.1:{port}/health").status_code == 200 for port in (18165, 18167))
    return {"all_passed": all(checks.values()), "checks": checks, "max_absolute_probability_delta": max_delta, "paired_questions": paired}


def summarize(report):
    rows, pairs = [], []
    for case in LABELS:
        for mode in MODES:
            ts = [t for t in report["trials"] if t["case"] == case and t["mode"] == mode and t["round"] > 0]
            r = {"case": case, "mode": mode, "repeats": len(ts), "min_seconds": min(t["wall_seconds"] for t in ts), "max_seconds": max(t["wall_seconds"] for t in ts)}
            for key in ("wall_seconds", "first_response_seconds", "last_response_seconds", "cpu_seconds", "cpu_mean_percent"):
                r[key] = statistics.median(t[key] for t in ts)
            for key in ("gpu_mean_percent", "rss_mean_gib", "system_cpu_mean_percent"):
                r[key] = statistics.median(t["resources"][key] for t in ts)
            for key in ("rss_peak_gib", "committed_peak_gib", "gpu_shared_peak_gib", "gpu_peak_percent"):
                r[key] = max(t["resources"][key] for t in ts)
            for phase in ("read_and_upload", "count_tokens", "predict", "remove_copy"):
                r[phase + "_seconds"] = statistics.median(sum(p["seconds"] for p in t["phases"] if p["kind"] == phase) for t in ts)
            r["cached_tokens_per_workflow"] = statistics.median(sum(a["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] for a in t["answers"]) for t in ts)
            rows.append(r)
        a, b = rows[-2:]
        rounds = []
        for index in (1, 2, 3):
            ts = {t["mode"]: t for t in report["trials"] if t["round"] == index and t["case"] == case}
            rounds.append({"round": index, "time_saved_percent": 100*(1-ts["optimized"]["wall_seconds"]/ts["baseline"]["wall_seconds"]),
                           "cpu_seconds_saved_percent": 100*(1-ts["optimized"]["cpu_seconds"]/ts["baseline"]["cpu_seconds"])})
        pairs.append({"case": case, "median_time_saved_percent": 100*(1-b["wall_seconds"]/a["wall_seconds"]),
                      "median_cpu_seconds_saved_percent": 100*(1-b["cpu_seconds"]/a["cpu_seconds"]), "paired_rounds": rounds})
    return {"statistic": "Three post-warmup workflows per case and mode; medians for times/means, maximum of the 3 runs for peaks", "rows": rows, "comparisons": pairs}


def render(report, summary, verification):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), layout="constrained")
    for ax, (field, title, unit) in zip(axes.flat, [("wall_seconds", "Tiempo para tres respuestas", "s"), ("cpu_seconds", "Trabajo total de CPU", "s de CPU"), ("committed_peak_gib", "Memoria privada comprometida: pico", "GiB"), ("gpu_mean_percent", "GPU: ocupación media del motor más activo", "%")]):
        for j, mode in enumerate(MODES):
            rs = [next(r for r in summary["rows"] if r["case"] == c and r["mode"] == mode) for c in LABELS]
            values = [r[field] for r in rs]
            errors = [[r[field]-r["min_seconds"] for r in rs], [r["max_seconds"]-r[field] for r in rs]] if field == "wall_seconds" else None
            bars = ax.bar([x+(j-.5)*.36 for x in range(3)], values, .34, label=MODES[mode], color=["#3d65bd", "#098477"][j], yerr=errors, capsize=3)
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_xticks(range(3), ["Imagen 45,9 MP", "Audio 30 s", "Vídeo 60 s"])
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, ax.get_ylim()[1]*1.12)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("E2B QAT Q4 · reutilización del archivo entre tres preguntas independientes")
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    f = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")
    lines = ["# E2B: reutilización del archivo entre preguntas", "", "Comparación emparejada del flujo anterior con la optimización. Mismos archivos públicos, preguntas, opciones y modelo. Se completaron 24 flujos y 72 inferencias; se excluyen 18 de calentamiento y se resumen tres repeticiones por caso y modo.", "",
             "| Archivo | Modo | Total 3 respuestas (s) | Rango (s) | Primera (s) | CPU media (%) | CPU acumulada (s) | RAM RSS pico (GiB) | Compromiso privado pico (GiB) | GPU media (%) | RAM compartida GPU pico (GiB) |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["rows"]:
        vals = [LABELS[r["case"]], MODES[r["mode"]], f(r["wall_seconds"]), f(r["min_seconds"])+"–"+f(r["max_seconds"]), f(r["first_response_seconds"]), f(r["cpu_mean_percent"], 1), f(r["cpu_seconds"]), f(r["rss_peak_gib"]), f(r["committed_peak_gib"]), f(r["gpu_mean_percent"], 1), f(r["gpu_shared_peak_gib"])]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "Los tiempos y las medias son medianas de tres pasadas. Los picos son el máximo de esas pasadas. CPU sobre 16 procesadores lógicos. GPU es el motor más activo, ponderado por tiempo.", "", "## Ahorro medido", "", "| Archivo | Ahorro de tiempo, medianas | Ahorro de CPU acumulada | Ahorro de tiempo por ronda |", "|---|---:|---:|---|"]
    for c in summary["comparisons"]:
        lines.append("| " + " | ".join([LABELS[c["case"]], f(c["median_time_saved_percent"], 1)+"%", f(c["median_cpu_seconds_saved_percent"], 1)+"%", "; ".join(f(p["time_saved_percent"], 1)+"%" for p in c["paired_rounds"])]) + " |")
    lines += ["", "## Qué cambia", "", "- Ambos modos suben una sola copia del archivo y hacen tres comprobaciones externas de tokens y tres inferencias secuenciales, con otra comprobación exacta interna por pregunta.",
              "- Anterior: reuse_media=false y cache_prompt=false. Reproduce la preparación y los mensajes de la versión archivada; repite la preparación en las seis operaciones de conteo/inferencia.",
              "- Optimizado: una preparación conservada en RAM (audio convertido, fotogramas o imagen en base64) y reutilizada por los seis pasos. El máximo contabilizado del contenido es 64 MiB, con descarte LRU y caducidad de cinco minutos sin acceso, comprobada en la siguiente operación. Borrar la copia cargada o cerrar el adaptador libera esa preparación.",
              "- Se valida la ruta y el tamaño y se calcula SHA-256 del archivo en cada operación. Cambiar archivo, modalidad, duración de audio o fotogramas invalida la preparación. Cada petición recibe su propia lista de contenido, sin acumular preguntas.",
              "- Primera inferencia de cada flujo con cache_prompt=false en ambos modos. La segunda y tercera optimizadas reutilizan el prefijo multimodal del motor. No se hereda una ventaja de la pasada anterior. El motor aún decodifica y tokeniza la entrada; evita reevaluar el prefijo ya calculado.",
              "- El payload conserva una sola pregunta con sus opciones. No se incluyen respuestas anteriores. Las probabilidades se calculan de nuevo y se normalizan por pregunta; siguen sin calibrar.",
              "- Contar tokens no ejecuta inferencia. Se sigue contando el prompt completo, incluidos los tokens reutilizados; el límite sigue siendo 4095 de entrada más uno de salida. Reutilizar un prefijo no amplía el contexto.", "",
              "## Verificación y alcance", "", f"Verificación independiente: {'CORRECTA' if verification['all_passed'] else 'FALLIDA'} ({len(verification['checks'])} comprobaciones). Las 36 parejas de preguntas conservaron la misma opción elegida. Máxima diferencia absoluta entre probabilidades: {f(100*verification['max_absolute_probability_delta'], 6)} puntos porcentuales; la caché nativa puede producir pequeñas diferencias numéricas.",
              "Se recalcularon RAM/GPU desde las muestras originales y CPU desde las instantáneas de Job Object de Windows. Incluye servicio, backend, procesos de conversión terminados y cliente/monitor; excluye navegador y procesos E4B. E4B permaneció cargado e inactivo. El modelo E2B ya estaba cargado; no se vació la caché de archivos del sistema operativo.",
              "La memoria comprometida puede estar respaldada por RAM o paginación; RSS mide páginas residentes. La GPU integrada comparte RAM: no sumar su cifra como memoria física independiente. Los picos entre rondas dependen también de asignaciones que el runtime conserva; una menor duración no implica menor porcentaje medio de GPU.",
              f"Muestreo real mediano: {f(report['sample_intervals']['median_seconds'], 3)} s; máximo: {f(report['sample_intervals']['max_seconds'], 3)} s. Otras aplicaciones siguieron activas. Los rangos describen esta sesión; no se hizo una prueba de significación estadística ni una evaluación de precisión con etiquetas humanas.",
              "Archivos: imagen de los Alpes de 10.109 × 4.542 px; audio de voz española de 30 s; vídeo de Marsella de 60 s a 16 fotogramas, sin su pista de audio. Procedencia y licencias: ../../real-media-benchmark-20261017/ATTRIBUTION.md.", "",
              "plan.json conserva el protocolo y las huellas; source/ archiva el adaptador medido; results.json conserva los resultados completos; samples.jsonl las lecturas originales; verification.json valida las parejas, presupuestos, cachés y recursos. El snapshot de la versión anterior está en ../baseline/.",
              "Fuentes técnicas: https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/README.md (cache_prompt y diferencias numéricas); https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-common.cpp#L743 (comparación de identidad y tamaño de los fragmentos multimodales). Las fuentes descargadas se contrastaron con el commit del motor instalado.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf8")
    body, table = [], False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            cells = [html.escape(v.strip()) for v in line.strip("|").split("|")]
            tag = "td" if table else "th"
            if not table:
                body.append('<div class="scroll"><table>')
                table = True
            body.append("<tr>" + "".join(f"<{tag}>{v}</{tag}>" for v in cells) + "</tr>")
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
    encoded = base64.b64encode((OUT / "comparison.png").read_bytes()).decode()
    body.insert(2, f'<img alt="Tiempo, CPU, memoria y GPU antes y después" src="data:image/png;base64,{encoded}">')
    nav = ' · '.join(f'<a href="{p}">{label}</a>' for p, label in (("summary.json", "Resumen"), ("trials.csv", "CSV"), ("verification.json", "Verificación"), ("results.json", "Resultados"), ("samples.jsonl", "Muestras")))
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>E2B: optimización multimodal</title><style>body{font:16px/1.55 system-ui;max-width:1320px;margin:2rem auto;padding:0 1.5rem;background:#f4f7f6;color:#19302e}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:14px}th,td{padding:.7rem;text-align:left;border-bottom:1px solid #d7e2de}th{background:#e3efeb}h2{margin-top:2rem}a{color:#076c63}</style><nav>' + nav + '</nav>' + '\n'.join(body) + '</html>'
    (OUT / "report.html").write_text(doc, encoding="utf8")


def main():
    report = json.loads((OUT / "results.json").read_text(encoding="utf8"))
    samples = [json.loads(line) for line in (OUT / "samples.jsonl").read_text(encoding="utf8").splitlines()]
    verification = verify(report, samples)
    save("verification.json", verification)
    assert verification["all_passed"], verification["checks"]
    summary = summarize(report)
    save("summary.json", summary)
    with (OUT / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        keys = ("case", "mode", "round", "wall_seconds", "first_response_seconds", "cpu_seconds", "cpu_mean_percent", "rss_peak_gib", "committed_peak_gib", "gpu_shared_peak_gib", "gpu_mean_percent")
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for t in report["trials"]:
            writer.writerow({k: t[k] if k in t else t["resources"][k] for k in keys})
    render(report, summary, verification)
    print(json.dumps({"verified": verification["all_passed"], "checks": len(verification["checks"]), "max_probability_delta": verification["max_absolute_probability_delta"], "summary": summary}, ensure_ascii=True))


if __name__ == "__main__":
    main()
