"""Auditable report for the third Spanish classifier evaluation."""
import base64
from collections import Counter
import html
from pathlib import Path

from .eval_report import figures, percent
from .focus_experiment import read_json


def render(run_dir):
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "summary.json")
    selection = read_json(run_dir / "selection.json")
    development = read_json(run_dir / "development-selection.json")
    manifest = read_json(run_dir.parents[1] / "manifest.json")
    cases = read_json(run_dir / "test-cases.json")
    before = summary["baseline"]["calibrated"]
    raw, after = summary["test"]["raw"], summary["test"]["calibrated"]
    paired = summary["paired_comparison"]
    focus = summary["focus"]
    figures(run_dir, summary, selected_title="Calibración seleccionada" if summary["validation_approved"] else "T=1 · nueva calibración rechazada")
    metrics_rows = []
    for label, key, fmt in (
        ("Exactitud", "accuracy", percent), ("Precisión macro", "macro_precision", percent),
        ("Recall macro", "macro_recall", percent), ("F1 macro", "macro_f1", percent),
        ("NLL ↓", "negative_log_loss", lambda x: f"{x:.4f}"),
        ("Brier multiclase (suma) ↓", "brier_multiclass_sum", lambda x: f"{x:.4f}"),
        ("ECE (10 intervalos) ↓", "ece_10_bins", percent),
        ("Errores con confianza ≥90%", "errors_with_confidence_ge_0_9", str),
    ):
        metrics_rows.append([label]+[fmt(item[key]) for item in (before, raw, after)])
    prior_classes = {row["label"]: row for row in before["per_class"]}
    current_classes = {row["label"]: row for row in after["per_class"]}
    focus_rows, all_rows = [], []
    for label, row in current_classes.items():
        old = prior_classes[label]
        all_rows.append([label, str(row["support"]), percent(old["f1"]), percent(row["f1"]),
                         percent(old["precision"]), percent(row["precision"]), percent(old["recall"]), percent(row["recall"]),
                         f"{100*(row['f1']-old['f1']):+.2f}"])
        if label in focus:
            lo, hi = paired["focus_f1_change_95ci"][label]
            focus_rows.append([label, str(row["support"]), percent(old["f1"]), percent(row["f1"]),
                               f"{100*(row['f1']-old['f1']):+.2f}", f"[{100*lo:+.2f}, {100*hi:+.2f}]"])
    lo, hi = paired["paired_bootstrap_95ci"]
    opening = (f"En {after['n']} mensajes nuevos, la exactitud pasa de {percent(before['accuracy'])} (v2) a "
               f"{percent(after['accuracy'])} (v3). Cambio: {100*paired['accuracy_gain']:+.2f} puntos porcentuales, "
               f"IC bootstrap pareado del 95% [{100*lo:+.2f}, {100*hi:+.2f}]. "
               f"Se corrigen {paired['fixed_errors']} errores y se introducen {paired['introduced_errors']}; "
               f"McNemar exacto p={paired['mcnemar_exact_pvalue']:.5g}.")
    sampling = ("Esta prueba contiene 50 mensajes de general, 50 de listas y 50 de noticias; los otros 450 se reparten "
                "proporcionalmente entre las otras 15 categorías. Las tres categorías prioritarias representan el 25% de la muestra, "
                "por diseño. Las métricas globales describen esta distribución enriquecida. La comparación v2/v3 usa exactamente "
                "los mismos mensajes y la misma evidencia de Gemma; no se compara directamente con el 88,44% histórico de otra muestra.")
    diagnosis = ("El análisis de los errores anteriores encontró tres fronteras difíciles: las afirmaciones conversacionales de general "
                 "se confunden con acciones o preguntas específicas; las operaciones sobre listas compiten con palabras temáticas como "
                 "contactos o cocina; noticias compite con recomendaciones y consultas de conocimiento. Se conservaron las etiquetas "
                 "originales, incluso cuando resultan ambiguas o poco intuitivas.")
    changes = [
        f"Se entrenó una regresión logística multiclase con TF-IDF de palabras y caracteres sobre {manifest['training_count']:,} mensajes del train oficial, reutilizando el banco congelado de v2.",
        "El apoyo solo se activa cuando Gemma o el clasificador auxiliar elige general, lists o news. En el resto de los casos se conserva la distribución calibrada de v2 antes de la nueva temperatura global.",
        f"Pesos del apoyo auxiliar: {summary['selected_weight']}. La categoría elegida por Gemma determina el peso si es prioritaria; en otro caso lo determina la categoría prioritaria del auxiliar.",
        "El entrenamiento pondera general ×3, noticias ×1,5 y listas ×1,2; las demás clases ×1. C=4. Los parámetros se seleccionaron en desarrollo, sin entrenar los pesos de Gemma.",
        "La mezcla es una media aritmética convexa de probabilidades. El componente Gemma usa su temperatura v2 congelada; una nueva temperatura escalar se aplica después a la mezcla completa.",
        "No hay una segunda llamada al LLM por mensaje: los resultados pareados reutilizan los logits reales de una misma inferencia, y las salidas derivadas quedan identificadas como tales en los JSONL.",
    ]
    dev = development["selected"]
    protocol = (f"Desarrollo: 300 mensajes nuevos de la validación oficial, con 25 por categoría prioritaria. "
                f"El F1 medio prioritario mejora {100*dev['mean_focus_f1_gain']:.2f} puntos en desarrollo. "
                "El criterio exigía ≥2 puntos de mejora media, ninguna caída prioritaria mayor de 1 punto y caídas globales "
                "de exactitud/F1 macro menores o iguales a 0,5 puntos. Se exploraron tres cabezales y 64 combinaciones de pesos "
                "por cabezal. La ampliación del plan de desarrollo está registrada antes de abrir el test. "
                "Desarrollo, calibración (300), validación (180) y prueba (600) son mutuamente disjuntos por ID y texto normalizado, "
                "quedan fuera del banco de entrenamiento y excluyen todas las muestras utilizadas en v1 y v2. "
                "El test anterior sirvió solo para diagnosticar errores, nunca como entrenamiento.")
    nll_change = summary["test"]["nll_change"]
    nll_lo, nll_hi = nll_change["paired_bootstrap_95ci"]
    calibration = (f"T candidata={summary['temperature']:.6f}; T aplicada={summary['selected_temperature']:.6f}. "
                   "La temperatura se ajustó con 300 mensajes mediante muestreo proporcional del remanente de validación y se aceptó únicamente si reducía "
                   "NLL en otros 180 mensajes de validación. "
                   f"NLL de validación: {selection['raw_validation']['negative_log_loss']:.4f} → "
                   f"{selection['candidate_validation']['negative_log_loss']:.4f}. "
                   f"Validación aprobada: {'sí' if summary['validation_approved'] else 'no'}. "
                   "La temperatura se congeló antes de consultar la prueba y conserva las categorías predichas. "
                   f"En el test, recalibrar cambia NLL en {nll_change['mean_calibrated_minus_raw']:+.4f}, "
                   f"IC95% [{nll_lo:+.4f}, {nll_hi:+.4f}]. "
                   "Como la calibración tiene proporciones naturales y la prueba prioriza tres clases, ECE/NLL del test deben "
                   "interpretarse junto con el muestreo y no como garantías para cualquier tráfico real.")
    residual = Counter((row["target"], row["v3_prediction"]) for row in cases
                       if not row["v3_correct"] and (row["target"] in focus or row["v3_prediction"] in focus))
    residual_rows = [[a,b,str(n)] for (a,b),n in residual.most_common()]
    fixed = [row for row in cases if row["v3_correct"] and not row["v2_correct"]]
    broken = [row for row in cases if row["v2_correct"] and not row["v3_correct"]]
    fixed.sort(key=lambda row: row["target"] not in focus)
    broken.sort(key=lambda row: row["target"] not in focus)
    limits = [
        "El corpus público MASSIVE evalúa 18 categorías de solicitudes de asistente en español. No valida otras taxonomías, mensajes de producción ni decisiones sobre imagen/audio/video.",
        f"{min(current_classes[c]['support'] for c in focus)} casos por categoría prioritaria dejan incertidumbre apreciable. Los intervalos de cambio F1 que incluyen cero no permiten afirmar una mejora concluyente de esa categoría.",
        "No se puede descartar que Gemma haya visto este corpus público durante su preentrenamiento. Paráfrasis relacionadas pueden hacer optimistas los intervalos que suponen mensajes independientes.",
        "General incluye peticiones heterogéneas y etiquetas ambiguas; se cuentan los errores originales. Un auxiliar léxico puede fallar con paráfrasis nuevas, contexto omitido o vocabulario fuera del dominio.",
        "ECE depende de los intervalos. La exactitud no cambia al recalibrar; la calibración puede mejorar NLL y empeorar ECE en una muestra concreta. Se muestran las tres métricas probabilísticas.",
        "Los perfiles están ligados por huellas al clasificador, banco, modelo y runtime evaluados. Cada perfil de calibración debe usarse con la tarea de su misma versión.",
    ]
    verification = run_dir / "verification.json"
    verification_text = "Verificación local pendiente."
    if verification.exists():
        proof = read_json(verification)
        verification_text = f"Verificación automática: {'correcta' if proof['passed'] else 'falló'}; {proof['unit_tests_passed']} pruebas unitarias."
        live = proof.get("live", {})
        if live:
            verification_text += f" API v3, fórmula de mezcla, calibración, rechazo de perfiles incompatibles y orden de opciones: {live.get('passed', False)}."
            delta = live.get("initial_to_warm_max_probability_delta", 0)
            limits.append(f"La repetición en Vulkan varió hasta {100*delta:.4f} puntos de probabilidad entre el primer prefill y la caché reutilizada. La prueba del orden compara consultas consecutivas del mismo contexto; no se promete igualdad bit a bit entre estados de caché.")
    usage = ('python -m jev_qat predict --input examples/decision-massive-v3-calibrated.json\n'
             'python scripts/verify_spanish_v3.py --live\n\n'
             'from jev_qat import classify_spanish_message\n'
             'result = classify_spanish_message("Añade arroz a mi lista de compras", version="v3")\n'
             'print(result["category"], result["category_probabilities"])')
    evidence = (f"Ejecución {summary['run_id']}. summary.json, test-cases.json, selection.json, development-selection.json, "
                "environment.json, source/, *-gemma.jsonl y *-predictions.jsonl conservan los resultados y sus huellas. "
                "Las salidas v3 derivadas se pueden recalcular sin volver a consultar el LLM. El código de verificación también "
                "contrasta esa fórmula con respuestas reales de la API.")
    sources = ('Datos originales CC BY 4.0, FitzGerald et al. / Amazon Science; el espejo MTEB declara Apache-2.0. '
               'Revisión de datos y SHA-256 fijados en los manifiestos. '
               '[MASSIVE](https://github.com/alexa/massive), '
               '[espejo MTEB](https://huggingface.co/datasets/mteb/amazon_massive_scenario), '
               '[temperature scaling](https://arxiv.org/abs/1706.04599).')
    edition = summary["schema_version"]
    if edition == 4:
        opening = opening.replace("(v3)", "(v4)")
        opening += " La mejora de un solo mensaje no demuestra una ganancia estable. V4 queda experimental y v2 calibrada continúa como versión predeterminada."
        sampling = ("La nueva prueba contiene 600 mensajes nunca usados en v1, v2 ni v3: 30 por cada categoría prioritaria "
                    "y otros 510 proporcionales de las 15 restantes. Las categorías prioritarias representan el 15% por diseño. "
                    "V2 y V4 se comparan sobre los mismos mensajes; las cifras de muestras históricas no son referencias directamente comparables.")
        changes = [
            f"Se conserva Gemma, su prompt v2 y el banco de {manifest['training_count']:,} ejemplos de entrenamiento. No se modifican sus pesos.",
            "El auxiliar usa regresión logística C=16 sobre TF-IDF de palabras y caracteres, sin ponderación adicional de clases. Se entrenó exclusivamente con el banco del train oficial.",
            f"La mezcla sólo se activa si el auxiliar elige general, listas o noticias con probabilidad ≥{percent(summary['selected_threshold'])}. Si no cumple ambas condiciones se conserva la distribución calibrada v2 antes de la temperatura final.",
            f"Peso del auxiliar: {percent(summary['selected_weight'])}; peso de Gemma: {percent(1-summary['selected_weight'])}. Nunca se usa una elección auxiliar ajena a las tres categorías para desplazar la decisión de Gemma.",
            "Las distribuciones se mezclan aritméticamente. La nueva temperatura se aplica al resultado completo. Los JSONL distinguen inferencias reales de Gemma y resultados derivados, que pueden recalcularse sin repetir el LLM.",
        ]
        protocol = ("La primera mezcla amplia (v3) se rechazó: en su prueba de 600 mensajes redujo exactitud de 89,17% a 88,33%, "
                    "aunque mejoró listas. Esa prueba se conservó como diagnóstico y quedó excluida de esta nueva prueba. "
                    "Se buscaron 48 alternativas conservadoras en 480 registros de desarrollo: los 300 de desarrollo v3 más los 180 "
                    "que habían validado su temperatura; estos últimos pasan explícitamente a ser desarrollo y no validan la calibración actual. "
                    f"El F1 medio prioritario de desarrollo mejora {100*dev['mean_focus_f1_gain']:.2f} puntos. "
                    "El criterio exigía ≥1 punto de mejora media, ninguna caída de F1 en las tres categorías, mejora global de exactitud y F1 macro "
                    "y exactitud no inferior en cada uno de los dos bloques. Se eligió mayor exactitud, luego F1 macro y luego umbral. "
                    "Los nuevos 300 registros de calibración, 150 de validación y 600 de prueba no comparten IDs ni textos normalizados "
                    "entre sí, con entrenamiento ni con ninguna muestra previa. No se reetiquetaron los errores.")
        calibration = calibration.replace("otros 180", "otros 150")
        if not summary["validation_approved"]:
            usage = ('python -m jev_qat predict --input examples/decision-massive-v4.json\n'
                     'python scripts/verify_spanish_v4.py --live\n\n'
                     'from jev_qat import classify_spanish_message\n'
                     'result = classify_spanish_message("Añade arroz a mi lista de compras", version="v4", calibrated=False)\n'
                     'print(result["category"], result["category_probabilities"])')
    metric_headers = ["Métrica", "V2 calibrada", "V3 antes de recalibrar", "V3 recalibrada"]
    if not summary["validation_approved"]:
        metric_headers[-1] = f"V{edition} seleccionada (T=1)"
    focus_headers = ["Categoría", "Casos", "F1 v2", "F1 v3", "Cambio (puntos)", "IC95% del cambio"]
    all_headers = ["Categoría", "Casos", "F1 v2", "F1 v3", "Precisión v2", "Precisión v3", "Recall v2", "Recall v3", "Cambio F1 (puntos)"]
    def md_table(headers, rows):
        clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        return "| "+" | ".join(map(clean, headers))+" |\n|"+"---|"*len(headers)+"\n"+"\n".join("| "+" | ".join(map(clean, row))+" |" for row in rows)
    def case_rows(values):
        return [[row["text"], row["target"], row["v2_prediction"], row["v3_prediction"]] for row in values[:12]]
    case_headers = ["Mensaje", "Etiqueta humana", "V2", "V3"]
    sections = [
        ("Diagnóstico", diagnosis), ("Prueba independiente", protocol), ("Recalibración", calibration),
    ]
    md = "# General, listas y noticias · evaluación v3\n\n"+opening+"\n\n"+sampling+"\n\n"+md_table(focus_headers, focus_rows)
    md += "\n\n"+md_table(metric_headers, metrics_rows)+"\n\n## Cambios\n\n"+"\n".join("- "+item for item in changes)
    md += "".join(f"\n\n## {title}\n\n{text}" for title,text in sections)
    md += "\n\n![Fiabilidad](reliability.png)\n\n## Todas las categorías\n\n"+md_table(all_headers, all_rows)
    md += "\n\n![Confusión](confusion.png)\n\n## Errores prioritarios restantes\n\n"+md_table(["Etiqueta humana", "Predicción v3", "Casos"], residual_rows)
    md += "\n\n## Errores corregidos\n\n"+md_table(case_headers, case_rows(fixed))+"\n\n## Errores introducidos\n\n"+md_table(case_headers, case_rows(broken))
    remaining = [row for row in cases if not row["v3_correct"] and row["target"] in focus]
    md += "\n\n## Ejemplos de errores restantes\n\n"+md_table(case_headers, case_rows(remaining))
    md += "\n\n## Uso\n\n```text\n"+usage+"\n```\n\n## Verificación\n\n"+verification_text
    md += "\n\n## Alcance\n\n"+"\n".join("- "+item for item in limits)+"\n\n## Evidencia\n\n"+evidence+"\n\n"+sources+"\n"
    if edition == 4:
        # Shared historical JSON case keys retain v3 for compatibility; the
        # rendered labels describe the actual current candidate.
        md = md.replace("F1 v3", "F1 v4").replace("V3", "V4").replace("Predicción v3", "Predicción v4").replace("API v3", "API v4")
        md = md.replace("evaluación v3", "evaluación v4").replace('version="v3"','version="v4"')
        md = md.replace("massive-v3", "massive-v4").replace("massive_es_v3", "massive_es_v4").replace("massive_scenario_es_v3", "massive_scenario_es_v4")
        md = md.replace("verify_spanish_v3", "verify_spanish_v4").replace("Precisión v3", "Precisión v4").replace("Recall v3", "Recall v4")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    escape = lambda value: html.escape(str(value))
    def table(headers, rows):
        return '<div class="table"><table><thead><tr>'+"".join("<th>"+escape(value)+"</th>" for value in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join("<td>"+escape(value)+"</td>" for value in row)+"</tr>" for row in rows)+"</tbody></table></div>"
    paragraph = lambda text: "<p>"+escape(text)+"</p>"
    listing = lambda items: "<ul>"+"".join("<li>"+escape(item)+"</li>" for item in items)+"</ul>"
    def plot(name):
        return f'<img alt="{escape(name)}" src="data:image/png;base64,{base64.b64encode((run_dir / name).read_bytes()).decode()}">'
    body = '<h1>General, listas y noticias · evaluación v3</h1>'+paragraph(opening)+paragraph(sampling)
    body += table(focus_headers, focus_rows)+table(metric_headers, metrics_rows)+"<h2>Cambios</h2>"+listing(changes)
    body += "".join("<h2>"+escape(title)+"</h2>"+paragraph(text) for title,text in sections)
    body += plot("reliability.png")+"<h2>Todas las categorías</h2>"+table(all_headers, all_rows)+plot("confusion.png")
    body += "<h2>Errores prioritarios restantes</h2>"+table(["Etiqueta humana", "Predicción v3", "Casos"], residual_rows)
    body += "<h2>Errores corregidos</h2>"+table(case_headers, case_rows(fixed))+"<h2>Errores introducidos</h2>"+table(case_headers, case_rows(broken))
    body += "<h2>Ejemplos de errores restantes</h2>"+table(case_headers, case_rows(remaining))
    api_usage = ('API: task_profile="massive_es_v3", calibration_profile="massive_scenario_es_v3"; la solicitud de ejemplo incluye las 18 opciones evaluadas.'
                 if summary["validation_approved"] else f'API: task_profile="{summary["task_profile"]}", sin calibration_profile. La nueva temperatura no fue validada; metadata.calibrated=false. El helper mantiene v2 como versión predeterminada.')
    body += "<h2>Uso</h2><pre>"+escape(usage)+"</pre>"+paragraph(api_usage)
    body += "<h2>Verificación</h2>"+paragraph(verification_text)+"<h2>Alcance</h2>"+listing(limits)+"<h2>Evidencia</h2>"+paragraph(evidence)
    body += '<p>Datos originales CC BY 4.0, FitzGerald et al. / Amazon Science; espejo Apache-2.0. <a href="https://github.com/alexa/massive">MASSIVE</a> · <a href="https://huggingface.co/datasets/mteb/amazon_massive_scenario">MTEB</a> · <a href="https://arxiv.org/abs/1706.04599">Guo et al.</a>. Ver manifiestos y NOTICE.</p>'
    if edition == 4:
        body = body.replace("F1 v3", "F1 v4").replace("V3", "V4").replace("evaluación v3", "evaluación v4")
        body = body.replace("Predicción v3", "Predicción v4").replace("API v3", "API v4")
        body = body.replace("Precisión v3", "Precisión v4").replace("Recall v3", "Recall v4")
        body = body.replace("massive-v3", "massive-v4").replace("massive_es_v3", "massive_es_v4").replace("massive_scenario_es_v3", "massive_scenario_es_v4")
        body = body.replace("verify_spanish_v3", "verify_spanish_v4").replace('version=&quot;v3&quot;', 'version=&quot;v4&quot;')
    (run_dir / "report.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>General, listas y noticias · evaluación v{edition}</title><style>body{{font:16px/1.55 system-ui,sans-serif;color:#182838;max-width:1150px;margin:40px auto;padding:0 24px}}'
        'h1{font-size:32px}h2{margin-top:40px}.table{overflow:auto}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:14px}'
        'th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e9}th{background:#eef3f7}img{width:100%;height:auto}a{color:#125c9a}'
        'li{margin-bottom:10px}pre{padding:16px;background:#eef3f7;overflow:auto}</style>'+body+'</html>', encoding="utf-8")
