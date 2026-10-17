"""Fresh paired comparison report for the improved Spanish task profile."""
import base64
import html
import json
from pathlib import Path

from .eval_report import figures, percent


def render(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    development = json.loads((run_dir / "development-selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((run_dir.parents[1] / "error-analysis.json").read_text(encoding="utf-8"))
    cases = json.loads((run_dir / "test-cases.json").read_text(encoding="utf-8"))
    before = summary["baseline"]["legacy_calibrated"]
    raw, after = summary["test"]["raw"], summary["test"]["calibrated"]
    paired = summary["paired_comparison"]
    figures(run_dir, summary)
    rows = []
    for title, key, fmt in (
        ("Exactitud (accuracy)", "accuracy", percent), ("Precisión macro", "macro_precision", percent),
        ("Recall macro", "macro_recall", percent), ("F1 macro", "macro_f1", percent),
        ("NLL ↓", "negative_log_loss", lambda x: f"{x:.4f}"),
        ("Brier multiclase (suma) ↓", "brier_multiclass_sum", lambda x: f"{x:.4f}"),
        ("ECE, 10 intervalos ↓", "ece_10_bins", percent),
        ("Confianza media", "mean_confidence", percent),
        ("Errores con confianza ≥90%", "errors_with_confidence_ge_0_9", str),
    ):
        rows.append([title, fmt(before[key]), fmt(raw[key]), fmt(after[key])])
    high_confidence = [next(row for row in item["coverage"] if row["threshold"] == .9) for item in (before, raw, after)]
    rows.append(["Casos con confianza ≥90%"]+[str(row["count"]) for row in high_confidence])
    rows.append(["Cobertura con confianza ≥90%"]+[percent(row["coverage"]) for row in high_confidence])
    rows.append(["Exactitud con confianza ≥90%"]+[percent(row["accuracy"]) if row["count"] else "—" for row in high_confidence])
    gain_lo, gain_hi = paired["paired_bootstrap_95ci"]
    acc_lo, acc_hi = after["accuracy_95ci"]
    opening = (
        f"La exactitud de la variante con ejemplos recuperados es {percent(after['accuracy'])}, frente a "
        f"{percent(before['accuracy'])} de la versión anterior en los mismos {after['n']} mensajes nuevos de prueba. "
        f"La ganancia es {100*paired['accuracy_gain']:.2f} puntos porcentuales, IC bootstrap pareado del 95% "
        f"[{100*gain_lo:.2f}, {100*gain_hi:.2f}]. El IC de Wilson de la exactitud nueva es [{percent(acc_lo)}, {percent(acc_hi)}]."
    )
    comparison = (
        f"Se corrigieron {paired['fixed_errors']} errores de la versión anterior y se introdujeron {paired['introduced_errors']} errores nuevos. "
        f"Prueba exacta de McNemar: p={paired['mcnemar_exact_pvalue']:.6g}. "
        "Las cifras principales comparan ambas versiones en esta nueva muestra, no el 79,8% histórico de otra muestra."
    )
    diagnostic_rows = [[row["target"], row["prediction"], str(row["count"])] for row in analysis["confusion_pairs"][:8]]
    diagnosis = (
        f"La prueba anterior contenía {analysis['errors']} errores. General tenía muchos falsos positivos: "
        "su descripción como categoría residual competía con correo, redes sociales, hogar conectado y consultas concretas. "
        "También se confundían información musical con reproducción, actualidad con conocimiento general y recomendaciones de lugares con transporte. "
        "La inspección detectó mensajes ambiguos y algunas etiquetas poco intuitivas; no se cambiaron etiquetas ni se eliminaron estos fallos."
    )
    changes = [
        "Se acotó la categoría general a conversación, bromas, personalidad y capacidades del asistente, evitando usarla como categoría residual.",
        "Se aclaró la taxonomía: contactos en email, cafetera y limpieza en iot, actualidad deportiva en news y lugares o actividades cercanas en recommendation.",
        f"Cada consulta recupera seis ejemplos similares de un banco de {manifest['splits']['retrieval_training']['count']:,} mensajes del train oficial, con un máximo de dos por categoría.",
        "La búsqueda combina TF-IDF de palabras y caracteres. Gemma sigue eligiendo la categoría con sus logits; la búsqueda aporta contexto y no sustituye el modelo.",
        "Las etiquetas internas siguen un orden fijo y luego se remapean al orden de las opciones recibidas. Los pesos de Gemma no se entrenaron ni modificaron.",
    ]
    protocol = (
        f"La variante se seleccionó exclusivamente con 150 mensajes de desarrollo: exactitud "
        f"{percent(development['baseline']['accuracy'])} → {percent(development['candidate']['accuracy'])}. "
        "El criterio previo exigía mejorar al menos un punto porcentual sin reducir F1 macro. "
        "Los 450 mensajes de la nueva prueba excluyen todos los mensajes de v1; el banco de ejemplos, desarrollo, "
        "calibración, validación y prueba no comparten IDs ni textos normalizados. El test anterior se usó solo como diagnóstico. "
        "Las etiquetas humanas no se envían en las consultas de prueba ni se reetiquetaron los errores."
    )
    calibration = (
        f"La nueva temperatura, T={summary['temperature']:.6f}, se ajustó con 300 mensajes independientes del banco de ejemplos. "
        f"En otros 150 mensajes de validación, NLL pasó de {selection['raw_validation']['negative_log_loss']:.4f} "
        f"a {selection['candidate_validation']['negative_log_loss']:.4f}. "
        +(f"El perfil {summary['profile_name']} superó el criterio de validación. " if summary['validation_approved'] else "La temperatura no superó la validación; se conserva T=1. ")+
        "La temperatura y la decisión de aceptarla quedaron fijadas antes de consultar la prueba final. La calibración conserva las predicciones. "
        "Para la referencia se usa la temperatura v1 ya existente, sin reajustarla con los nuevos datos. "
        "Los errores por encima del 90% de confianza se muestran junto con su cobertura: las versiones envían cantidades distintas de mensajes a ese rango. "
        "Estos umbrales son descriptivos y no se eligió una regla de producción usando el test."
    )
    nll = summary["test"]["nll_change"]
    nll_ci = nll["paired_bootstrap_95ci"]
    calibration += (f" En la nueva prueba, el cambio de NLL al calibrar v2 fue {nll['mean_calibrated_minus_raw']:.4f}, "
                    f"IC bootstrap pareado del 95% [{nll_ci[0]:.4f}, {nll_ci[1]:.4f}].")
    audit = summary["order_audit"]
    order = (f"Al invertir las opciones en {audit['n']} mensajes, cambiaron {audit['prediction_disagreements']} predicciones. "
             f"La variación total media de las probabilidades fue {audit['mean_total_variation']:.6f}. "
             "La calibración, el banco y las instrucciones están ligados por huellas al perfil evaluado.")
    limits = [
        "El resultado se limita a las 18 categorías de solicitudes de asistente MASSIVE en español. No demuestra una mejora universal en mensajes de producción, imagen, audio o video.",
        "Algunos ejemplos tienen etiquetas ambiguas o requieren contexto ausente. Se conservaron sus etiquetas originales y se contabilizaron los fallos.",
        "Es un corpus público y no se puede descartar su presencia en el entrenamiento de Gemma. Los intervalos asumen mensajes independientes; paráfrasis relacionadas pueden hacerlos optimistas.",
        "Algunas categorías tienen pocos casos. El soporte se muestra junto al F1. ECE depende de los intervalos y se acompaña de NLL y Brier.",
        "El perfil se solicita explícitamente y necesita estas opciones y pregunta. El perfil de calibración v1 no es compatible con el nuevo prompt ni con otro banco de ejemplos.",
    ]
    verification_path = run_dir / "verification.json"
    if verification_path.exists():
        live = json.loads(verification_path.read_text(encoding="utf-8")).get("live", {})
        if "initial_to_warm_max_probability_delta" in live:
            delta = live["initial_to_warm_max_probability_delta"]
            limits.append(f"En una comprobación de repetición, el backend Vulkan varió hasta {100*delta:.3f} puntos porcentuales de probabilidad entre el primer prefill y la caché reutilizada. La comparación del orden se hace con consultas consecutivas del mismo contexto; no se garantiza igualdad bit a bit entre estados de caché.")
    per_class = [[a["label"], str(a["support"]), percent(b["f1"]), percent(a["f1"]), f"{100*(a['f1']-b['f1']):+.2f}"]
                 for a,b in zip(after["per_class"], before["per_class"])]
    headers = ["Métrica", "Anterior + calibración v1", "V2 sin calibrar", "V2 recalibrada"]
    def markdown_table(headers, rows):
        return "| "+" | ".join(headers)+" |\n|"+"---|"*len(headers)+"\n"+"\n".join("| "+" | ".join(row)+" |" for row in rows)
    usage = ("```powershell\npython -m jev_qat start\n"
             "python -m jev_qat predict --input examples/decision-massive-v2-calibrated.json\n"
             "python -m jev_qat.improve_spanish run\n"
             "python scripts/verify_spanish_v2.py --live\n```\n\n"
             "En la API o en Python se añaden `task_profile=\"massive_es_v2\"` y "
             "`calibration_profile=\"massive_scenario_es_v2\"` a la solicitud de 18 opciones. "
             "Sin el perfil de tarea se conserva el clasificador genérico. La ejecución se reanuda desde sus predicciones; no representa una nueva medición.")
    sources = ("[MASSIVE](https://github.com/alexa/massive), "
               "[copia MTEB](https://huggingface.co/datasets/mteb/amazon_massive_scenario), "
               "[FitzGerald et al.](https://arxiv.org/abs/2204.08582), "
               "[SLURP](https://aclanthology.org/2020.emnlp-main.588/), "
               "[temperature scaling](https://arxiv.org/abs/1706.04599). "
               "Datos originales CC BY 4.0, FitzGerald et al. / Amazon Science. El espejo declara Apache-2.0. "
               "La revisión y SHA-256 de origen se conservan en el manifiesto v1; el manifiesto v2 lo referencia por su SHA-256 y fija los nuevos subconjuntos y el banco.")
    md = (f"# Mejora del clasificador de mensajes en español\n\n{opening}\n\n{comparison}\n\n"+
          markdown_table(headers, rows)+f"\n\n## Diagnóstico inicial\n\n{diagnosis}\n\n"+
          markdown_table(["Etiqueta humana", "Predicción incorrecta", "Casos"], diagnostic_rows)+
          "\n\n## Cambios\n\n"+"\n".join("- "+text for text in changes)+
          f"\n\n## Desarrollo y prueba independiente\n\n{protocol}\n\n## Recalibración\n\n{calibration}\n\n"
          "![Fiabilidad](reliability.png)\n\n## Resultados por categoría\n\n"+
          markdown_table(["Categoría", "Casos", "F1 anterior", "F1 v2", "Cambio en puntos"], per_class)+
          f"\n\n![Confusión v2](confusion.png)\n\n## Orden de las opciones\n\n{order}\n\n## Uso y reproducción\n\n{usage}\n\n"
          "## Alcance\n\n"+"\n".join("- "+text for text in limits)+
          f"\n\n## Evidencia\n\n{sources}\n\n"
          f"Latencia mediana v1: {summary['latency_seconds']['baseline_median']:.3f} s; v2: {summary['latency_seconds']['v2_median']:.3f} s. "
          f"Ejecución {summary['run_id']}. Ver summary.json, test-cases.json, selection.json, development-selection.json, "
          "environment.json, source/ y las predicciones JSONL.\n")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    def table(headers, rows):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(value))}</th>" for value in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in row)+"</tr>" for row in rows)+"</tbody></table>"
    def plot(name):
        data = base64.b64encode((run_dir / name).read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{data}">'
    def paragraphs(text):
        return "<p>"+html.escape(text)+"</p>"
    def listing(values):
        return "<ul>"+"".join("<li>"+html.escape(value)+"</li>" for value in values)+"</ul>"
    fixed = [row for row in cases if row["v2_correct"] and not row["baseline_correct"]]
    broken = [row for row in cases if row["baseline_correct"] and not row["v2_correct"]]
    def case_table(values):
        return table(["Mensaje", "Etiqueta humana", "Anterior", "V2"],
                     [[row["text"], row["target"], row["baseline_prediction"], row["v2_prediction"]] for row in values[:12]])
    body = ("<h1>Mejora del clasificador de mensajes en español</h1>"+paragraphs(opening)+paragraphs(comparison)+table(headers, rows)+
            "<h2>Diagnóstico inicial</h2>"+paragraphs(diagnosis)+table(["Etiqueta humana", "Predicción incorrecta", "Casos"], diagnostic_rows)+
            "<h2>Cambios</h2>"+listing(changes)+"<h2>Desarrollo y prueba independiente</h2>"+paragraphs(protocol)+
            "<h2>Recalibración</h2>"+paragraphs(calibration)+plot("reliability.png")+
            "<h2>Resultados por categoría</h2>"+table(["Categoría", "Casos", "F1 anterior", "F1 v2", "Cambio en puntos"], per_class)+plot("confusion.png")+
            "<h2>Ejemplos de errores corregidos</h2>"+case_table(fixed)+"<h2>Ejemplos de errores nuevos</h2>"+case_table(broken)+
            "<h2>Orden de las opciones</h2>"+paragraphs(order)+"<h2>Usar el perfil</h2>"+
            '<pre>python -m jev_qat predict --input examples/decision-massive-v2-calibrated.json</pre>'+
            '<p>Campos: <code>task_profile="massive_es_v2"</code> y <code>calibration_profile="massive_scenario_es_v2"</code>. La consulta de ejemplo contiene las 18 opciones y la pregunta evaluadas.</p>'+
            "<h2>Alcance</h2>"+listing(limits)+
            '<h2>Fuentes</h2><p><a href="https://github.com/alexa/massive">MASSIVE</a> · <a href="https://huggingface.co/datasets/mteb/amazon_massive_scenario">MTEB</a> · '
            '<a href="https://arxiv.org/abs/1706.04599">Guo et al.</a>. Datos originales CC BY 4.0, FitzGerald et al. / Amazon Science; espejo Apache-2.0. Ver manifiestos y NOTICE.</p>'+
            paragraphs(f"Latencia mediana v1: {summary['latency_seconds']['baseline_median']:.3f} s; v2: {summary['latency_seconds']['v2_median']:.3f} s. Ejecución {summary['run_id']}."))
    (run_dir / "report.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Mejora de clasificación en español · E4B QAT</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#182838;max-width:1150px;margin:40px auto;padding:0 24px}'
        'h1{font-size:32px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e9}'
        'th{background:#eef3f7}img{width:100%;height:auto}a{color:#125c9a}li{margin-bottom:10px}pre{padding:16px;background:#eef3f7;overflow:auto}</style>'+body+'</html>', encoding="utf-8")
