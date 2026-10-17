"""Offline HTML/Markdown reports and exportable scientific plots."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .eval_metrics import wilson_interval


def percent(value):
    return f"{100*value:.2f}%"


def render(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    errors = json.loads((run_dir / "test-errors.json").read_text(encoding="utf-8"))
    raw, calibrated = summary["test"]["raw"], summary["test"]["calibrated"]
    figures(run_dir, summary)
    comparison = [
        ("Exactitud (accuracy)", percent(raw["accuracy"]), percent(calibrated["accuracy"])),
        ("Precisión macro", percent(raw["macro_precision"]), percent(calibrated["macro_precision"])),
        ("F1 macro", percent(raw["macro_f1"]), percent(calibrated["macro_f1"])),
        ("NLL ↓", f"{raw['negative_log_loss']:.4f}", f"{calibrated['negative_log_loss']:.4f}"),
        ("Brier multiclase (suma) ↓", f"{raw['brier_multiclass_sum']:.4f}", f"{calibrated['brier_multiclass_sum']:.4f}"),
        ("ECE, 10 intervalos ↓", percent(raw["ece_10_bins"]), percent(calibrated["ece_10_bins"])),
        ("Confianza media", percent(raw["mean_confidence"]), percent(calibrated["mean_confidence"])),
        ("Errores con confianza ≥90%", str(raw["errors_with_confidence_ge_0_9"]), str(calibrated["errors_with_confidence_ge_0_9"])),
    ]
    lo, hi = raw["accuracy_95ci"]
    delta = summary["test"]["nll_change"]
    delta_lo, delta_hi = delta["paired_bootstrap_95ci"]
    audit = summary["order_audit"]
    approved = summary["validation_approved"]
    decision = (f"Perfil validado: `{summary['profile_name']}`. Se activa solo al solicitarlo explícitamente."
                if approved else "La calibración no superó la validación. No se activa un perfil; la salida seleccionada conserva T=1.")
    explanation = (
        f"Se evaluaron 18 categorías con {raw['n']} mensajes de prueba sin usar para ajustar ni seleccionar la calibración. "
        f"La exactitud fue {percent(raw['accuracy'])}, IC del 95% de Wilson [{percent(lo)}, {percent(hi)}]. "
        "La temperatura mantiene la clase elegida; ajusta la distribución de probabilidades."
    )
    protocol = (
        f"Muestreo estratificado proporcional: 300 ejemplos del train oficial para calibrar, 150 del validation oficial para validar "
        f"y 450 del test oficial para la medición final. Semilla {manifest['protocol']['seed']}. "
        "Se eliminaron grupos de texto con etiquetas contradictorias y duplicados normalizados entre particiones. "
        "Las opciones se barajaron de forma determinista por mensaje y la respuesta humana nunca se envió al modelo. "
        f"T={summary['temperature']:.6f} se ajustó minimizando NLL en calibración. El criterio fijado antes de inferir fue mejorar "
        "la NLL de validación más de 0,0001. La prueba final no intervino en esta decisión."
    )
    validation = (
        f"NLL en validación: {selection['raw_validation']['negative_log_loss']:.4f} original → "
        f"{selection['candidate_validation']['negative_log_loss']:.4f} con la temperatura candidata. "
        f"Cambio de NLL en prueba (calibrada − original): {delta['mean_calibrated_minus_raw']:.4f}, "
        f"IC bootstrap pareado del 95% [{delta_lo:.4f}, {delta_hi:.4f}], {delta['repetitions']} remuestreos."
    )
    order = (
        f"En {audit['n']} mensajes de prueba repetidos con otro orden de opciones, cambiaron "
        f"{audit['prediction_disagreements']} predicciones ({percent(audit['prediction_disagreements']/audit['n'])}). "
        f"Exactitud en esa submuestra: {percent(audit['original_accuracy'])} → {percent(audit['reordered_accuracy'])}. "
        f"Distancia de variación total media entre distribuciones: {audit['mean_total_variation']:.4f}. "
        "Estas repeticiones no se suman al test ni se usan para ajustar T."
    )
    limits = [
        "MASSIVE contiene solicitudes de asistente localizadas por humanos. No representa todos los mensajes de redes sociales ni el tráfico real de tu aplicación.",
        "Es un benchmark público: no se puede descartar que el modelo haya visto sus datos durante el entrenamiento.",
        "450 ejemplos permiten una primera medición; algunas categorías tienen pocos casos. Consulta sus soportes antes de comparar F1.",
        "Los intervalos tratan cada mensaje como unidad independiente; la presencia de paráfrasis relacionadas puede hacerlos optimistas.",
        "La calibración solo se valida para texto en español con esta pregunta y estas 18 opciones. Imágenes, audio, video y otras taxonomías requieren evaluaciones propias.",
        "ECE depende de los intervalos elegidos. Se acompaña de NLL y Brier; valores menores son mejores en las tres métricas.",
        "T escalar no corrige errores de clasificación ni dependencia del orden. Los umbrales de confianza de este informe son descriptivos, no reglas de producción validadas.",
    ]
    table_md = "| Métrica | Original | Calibrada seleccionada |\n|---|---:|---:|\n"+"\n".join(f"| {name} | {a} | {b} |" for name, a, b in comparison)
    coverage = []
    for row in calibrated["coverage"]:
        interval = (f"[{percent(row['accuracy_95ci'][0])}, {percent(row['accuracy_95ci'][1])}]" if row["count"] else "—")
        coverage.append([percent(row["threshold"]), str(row["count"]), percent(row["coverage"]),
                         percent(row["accuracy"]) if row["count"] else "—", interval])
    coverage_md = "| Confianza mínima | Casos | Cobertura | Exactitud | IC 95% |\n|---|---:|---:|---:|---|\n"+"\n".join("| "+" | ".join(row)+" |" for row in coverage)
    per_class_md = "| Categoría | Casos | Precisión | Recall | F1 |\n|---|---:|---:|---:|---:|\n"+"\n".join(
        f"| {row['label']} | {row['support']} | {percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} |" for row in raw["per_class"])
    sources = (
        "[MASSIVE original](https://github.com/alexa/massive), "
        "[copia MTEB utilizada](https://huggingface.co/datasets/mteb/amazon_massive_scenario), "
        "[artículo MASSIVE](https://arxiv.org/abs/2204.08582), "
        "[SLURP: Bastianelli et al.](https://aclanthology.org/2020.emnlp-main.588/), "
        "[temperature scaling: Guo et al.](https://arxiv.org/abs/1706.04599). "
        "Datos originales: CC BY 4.0, FitzGerald et al. / Amazon Science; la ficha del espejo declara Apache-2.0. "
        f"Revisión de datos: `{manifest['revision']}`. Los SHA-256 y la atribución están en `manifest.json`."
    )
    usage = (
        "```powershell\npython -m jev_qat start\npython -m jev_qat.evaluate prepare\npython -m jev_qat.evaluate run\n"
        "python -m jev_qat.evaluate report\n"
        +(f"python -m jev_qat predict --input examples/decision-massive-calibrated.json\n" if approved else "")+"```"
    )
    content = (f"# Evaluación de clasificación en español\n\n{explanation}\n\n{decision}\n\n{table_md}\n\n"
               f"{validation}\n\n## Calibración y particiones\n\n{protocol}\n\n"
               f"## Fiabilidad de las probabilidades\n\n![Fiabilidad](reliability.png)\n\n"
               f"### Cobertura según confianza calibrada\n\n{coverage_md}\n\nEstos umbrales son descriptivos; no se eligió una regla de producción con la prueba final.\n\n"
               f"## Sensibilidad al orden\n\n{order}\n\n"
               f"## Resultados por categoría\n\n{per_class_md}\n\n![Matriz de confusión](confusion.png)\n\n"
               f"## Límites\n\n"+"\n".join(f"- {value}" for value in limits)+
               f"\n\n## Reproducir y usar\n\n{usage}\n\n"
               "La ejecución se reanuda desde las predicciones guardadas. Para una medición completamente nueva, usa "
               "`--directory evaluation/massive-es-repeticion` con `prepare` y `run`. El perfil no se aplica por defecto: "
               "se solicita mediante `calibration_profile` y conserva `raw_probabilities` para auditoría.\n\n"
               f"## Fuentes y trazabilidad\n\n{sources}\n\n"
               f"Ejecución `{summary['run_id']}`. Latencia mediana {summary['latency_seconds']['median']:.3f} s; "
               f"p95 {summary['latency_seconds']['p95']:.3f} s. Ver `summary.json`, `selection.json`, "
               "`environment.json`, `test-errors.json` y los cuatro archivos de predicciones JSONL.\n")
    (run_dir / "report.md").write_text(content, encoding="utf-8")
    def table(headers, rows):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(value))}</th>" for value in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in row)+"</tr>" for row in rows)+"</tbody></table>"
    def plot(name):
        data = base64.b64encode((run_dir / name).read_bytes()).decode()
        return f'<img alt="{name}" src="data:image/png;base64,{data}">'
    body = (f"<h1>Evaluación de clasificación en español</h1><p>{html.escape(explanation)}</p>"
            f"<p><strong>{html.escape(decision.replace('`', ''))}</strong></p>"+table(["Métrica", "Original", "Calibrada seleccionada"], comparison)+
            f"<p>{html.escape(validation)}</p><h2>Protocolo y calibración</h2><p>{html.escape(protocol)}</p>"+plot("reliability.png")+
            "<h2>Cobertura según confianza calibrada</h2>"+table(["Confianza mínima", "Casos", "Cobertura", "Exactitud", "IC 95%"], coverage)+
            "<p>Umbrales descriptivos; no se eligió una regla de producción con la prueba final.</p>"+
            f"<h2>Sensibilidad al orden</h2><p>{html.escape(order)}</p><h2>Resultados por categoría</h2>"+
            table(["Categoría", "Casos", "Precisión", "Recall", "F1"], [[row['label'], row['support'], percent(row['precision']), percent(row['recall']), percent(row['f1'])] for row in raw["per_class"]])+plot("confusion.png")+
            "<h2>Errores con mayor confianza original</h2>"+table(["Mensaje", "Etiqueta humana", "Predicción", "Original", "Calibrada"],
            [[row['text'], row['target'], row['prediction'], percent(row['raw_confidence']), percent(row['calibrated_confidence'])] for row in errors[:15]])+
            "<h2>Límites</h2><ul>"+"".join(f"<li>{html.escape(value)}</li>" for value in limits)+"</ul>"
            '<h2>Archivos</h2><p>Informe Markdown, summary.json, selection.json, environment.json, test-errors.json y predicciones JSONL junto a este HTML. Fuentes: '
            '<a href="https://github.com/alexa/massive">MASSIVE</a>, <a href="https://huggingface.co/datasets/mteb/amazon_massive_scenario">MTEB</a>, '
            '<a href="https://arxiv.org/abs/1706.04599">Guo et al.</a></p>'
            f"<p>Datos originales CC BY 4.0; FitzGerald et al. / Amazon Science. Revisión {manifest['revision']}. Ejecución {summary['run_id']}.</p>")
    (run_dir / "report.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Evaluación MASSIVE español · E4B QAT</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#182838;max-width:1100px;margin:40px auto;padding:0 24px}'
        'h1{font-size:32px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e9}'
        'th{background:#eef3f7}img{width:100%;height:auto}a{color:#125c9a}li{margin-bottom:10px}</style>'+body+'</html>', encoding="utf-8")


def figures(run_dir, summary, selected_title="Calibración seleccionada"):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    for column, (name, title, color) in enumerate((("raw", "Probabilidades originales", "#b05a12"), ("calibrated", selected_title, "#145c94"))):
        bins = summary["test"][name]["reliability"]
        present = [row for row in bins if row["count"]]
        x = np.array([row["confidence"] for row in present])
        y = np.array([row["accuracy"] for row in present])
        ci = np.array([wilson_interval(round(row["accuracy"]*row["count"]), row["count"]) for row in present])
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], "--", color="#87919a", label="Confianza = exactitud")
        ax.errorbar(x, y, yerr=np.maximum([y-ci[:, 0], ci[:, 1]-y], 0), fmt="none", ecolor=color, capsize=4, alpha=.6)
        ax.scatter(x, y, s=[30+row["count"] for row in present], color=color, zorder=3)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title, ylabel="Exactitud observada", xlabel="Confianza media por intervalo")
        ax.grid(alpha=.15)
        ax.legend(loc="upper left", fontsize=9)
        axes[1, column].bar([(row["lower"]+row["upper"])/2 for row in bins], [row["count"] for row in bins], width=.085, color=color)
        axes[1, column].set(xlabel="Confianza de la clase elegida", ylabel="Número de mensajes", xlim=(0, 1))
    fig.suptitle(f"Fiabilidad · MASSIVE español · {summary['test']['raw']['n']} mensajes de prueba\nIntervalos de Wilson al 95%; tamaño del punto proporcional al soporte", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(run_dir / f"reliability.{extension}", dpi=170)
    plt.close(fig)
    matrix = np.array(summary["test"]["raw"]["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(matrix)), yticks=range(len(matrix)), xticklabels=summary["classes"], yticklabels=summary["classes"],
           xlabel="Categoría predicha", ylabel="Categoría humana", title="Matriz de confusión · color normalizado por fila, números = casos")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            if matrix[row, column]:
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=9,
                        color="white" if normalized[row, column] > .5 else "#152538")
    fig.colorbar(image, ax=ax, shrink=.8, label="Proporción por categoría humana")
    for extension in ("png", "svg"):
        fig.savefig(run_dir / f"confusion.{extension}", dpi=170)
    plt.close(fig)
