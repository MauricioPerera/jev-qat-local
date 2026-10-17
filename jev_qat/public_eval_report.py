"""Self-contained report for the preregistered frozen-v8 follow-up."""
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .focus_experiment import read_json
from .public_eval_run import general


def table(headers, rows):
    cells = lambda row, tag: "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in row) + "</tr>"
    return "<div class='scroll'><table><thead>" + cells(headers, "th") + "</thead><tbody>" + "".join(cells(r, "td") for r in rows) + "</tbody></table></div>"


def render(run_dir):
    run_dir = Path(run_dir)
    s = read_json(run_dir / "summary.json")
    m = read_json(run_dir.parents[1] / "manifest.json")
    a = read_json(run_dir / "activation.json")
    cases = read_json(run_dir / "test-cases.json")
    b, c = s["baseline"]["calibrated"], s["test"]["calibrated"]
    paired = s["paired_comparison"]
    pct = lambda x: f"{100*x:.2f}%"
    fp = lambda value: int(np.array(value["confusion_matrix"])[:, 6].sum() - value["confusion_matrix"][6][6])
    significant = a["pvalue"] < a["alpha"]
    status = ("V8 activada: supera todos los criterios y la verificación." if a["activated"] else
              "V8 pendiente de verificación final; v2 sigue activa." if a["statistical_and_quality_gates_passed"] and a["verification_required"] else
              "V8 no se activa. V2 calibrada sigue activa.")
    opening = (f"Prueba independiente de {c['n']} mensajes nuevos: exactitud {pct(b['accuracy'])} → {pct(c['accuracy'])}. "
        f"{paired['fixed_errors']} errores corregidos y {paired['introduced_errors']} introducidos. "
        f"McNemar exacto unilateral p={a['pvalue']:.6g}; umbral preregistrado p < {a['alpha']}. "
        f"Mejora estadísticamente significativa en esta cohorte: {'sí' if significant else 'no'}.")
    data = (f"{m['counts']['datasets']['DIHANA']} cierres de conversación de DIHANA y {m['counts']['datasets']['MTOP']} solicitudes especializadas de MTOP. "
        "DIHANA fue anotado por personas y revisado por un anotador humano según sus autores. Se verificó cada segmento seleccionado "
        "contra U:Cierre:Nil:Nil del archivo original, con un único segmento por participante. MTOP conserva intenciones humanas y "
        "traducción profesional al español; cada texto, ID e intención se contrastaron contra el archivo oficial de Meta. "
        "Los IDs y las familias de texto normalizadas, ignorando puntuación, no aparecen en el entrenamiento ni en evaluaciones anteriores. "
        "El mapeo a nuestras categorías es determinista, congelado antes de inferir y sin una nueva adjudicación humana de ese mapeo.")
    stats = ("V8, sus umbrales y sus probabilidades permanecieron congelados. Esta cohorte tiene tamaño fijo y una única decisión final. "
        "Se usa p < 0,025 como umbral conservador para esta segunda evaluación de v8; la anterior dio p=0,5 y no se mezcla con la nueva "
        "para calcular significación. Esto no modifica retrospectivamente el procedimiento estadístico histórico. "
        "No se añaden ni reemplazan casos tras ver las predicciones. Los intervalos bootstrap al 95% son descriptivos.")
    scope = ("La prueba cubre general mediante cierres conversacionales y siete categorías especializadas de MTOP. "
        "No mide toda la variedad de general, las 18 categorías completas ni el tráfico real de producción. "
        "Las ocho recuperaciones anteriores son comprobaciones separadas; no se cuentan como ejemplos nuevos ni aumentan la significación. "
        "Una prueba sin significación no demuestra igualdad entre versiones. La exposición de Gemma a estos corpus públicos durante su preentrenamiento es desconocida. "
        "En MTOP no se publican IDs de los autores de cada mensaje; no puede descartarse dependencia residual entre plantillas o autores.")
    calibration = (f"Temperaturas conservadas: v2={s['frozen_temperatures']['v2']:.6f}, v8={s['frozen_temperatures']['v8']:.6f}. "
        "No se reajustó la calibración con esta prueba confirmatoria. NLL, Brier y fiabilidad evalúan la calibración existente en datos nuevos. "
        "Cambiar parámetros después de ver esta cohorte requeriría otra prueba independiente.")
    values = [[title, fmt(b[key]), fmt(c[key])] for title, key, fmt in (
        ("Exactitud", "accuracy", pct), ("F1 macro (18 categorías, incluidas sin soporte)", "macro_f1", pct),
        ("NLL ↓", "negative_log_loss", lambda x: f"{x:.4f}"),
        ("Brier (suma) ↓", "brier_multiclass_sum", lambda x: f"{x:.4f}"), ("ECE ↓", "ece_10_bins", pct))]
    gr = [[title, fmt(general(b)[key]), fmt(general(c)[key])] for title, key, fmt in (
        ("Precisión", "precision", pct), ("Recall", "recall", pct), ("F1", "f1", pct), ("Soporte", "support", str))]
    gr.append(["Falsos positivos general", fp(b), fp(c)])
    changed = [r for r in cases if r["v2_prediction"] != r["v8_prediction"]]
    regressions = read_json(run_dir / "regressions.json")
    # A small fixed chart directly compares existing probability calibration on fresh data.
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, value, title, color in zip(axes, (b, c), ("v2 calibrada", "v8 calibrada congelada"), ("#54697d", "#166b80")):
        rr = [r for r in value["reliability"] if r["count"]]
        ax.plot([0, 1], [0, 1], "--", color="#999999")
        ax.scatter([r["confidence"] for r in rr], [r["accuracy"] for r in rr], s=[r["count"] + 35 for r in rr], color=color)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Confianza media", ylabel="Exactitud observada", title=title)
        ax.grid(alpha=.15)
    fig.suptitle(f"Probabilidades en datos nuevos · n={c['n']} · sin reajustar temperaturas")
    fig.savefig(run_dir / "reliability.png", dpi=160)
    fig.savefig(run_dir / "reliability.svg")
    plt.close(fig)
    image = base64.b64encode((run_dir / "reliability.png").read_bytes()).decode()
    body = "<h1>Evaluación ampliada de v8 frente a v2</h1><div class='status'>" + html.escape(status) + "</div>"
    para = lambda text: "<p>" + html.escape(text) + "</p>"
    body += para(opening) + table(["Métrica", "v2", "v8"], values)
    body += "<h2>General y aciertos protegidos</h2>" + table(["Métrica", "v2", "v8"], gr)
    body += para(f"Aciertos recuperados conservados: {sum(r['passed'] for r in regressions['cases'][:8])}/8. "
                 f"Correcciones de dominio conservadas: {sum(r['passed'] for r in regressions['cases'][8:])}/3. "
                 "Se conserva la primera comprobación: un fallo impide activar v8 aunque la prueba nueva mejore.")
    body += "<h2>Etiquetas humanas documentadas y datos nuevos</h2>" + para(data)
    body += table(["Fuente", "Mensajes", "Exactitud v2", "Exactitud v8"], [[name, v["candidate"]["n"], pct(v["baseline"]["accuracy"]), pct(v["candidate"]["accuracy"])] for name, v in s["per_source"].items()])
    body += "<h2>Prueba estadística fijada antes de inferir</h2>" + para(stats)
    body += para("Cambio de exactitud e intervalo bootstrap descriptivo al 95%: " + pct(paired["accuracy_gain"]) + " [" + ", ".join(pct(x) for x in paired["paired_bootstrap_95ci"]) + "].")
    body += table(["Criterio obligatorio", "Cumple"], [[k, "Sí" if v else "No"] for k, v in a["conditions"].items()])
    body += "<h2>Probabilidades</h2>" + para(calibration) + f"<img alt='Fiabilidad de v2 y v8 en datos nuevos' src='data:image/png;base64,{image}'>"
    body += "<h2>Alcance de la evidencia</h2>" + para(scope)
    body += "<details><summary>Decisiones modificadas</summary>" + table(["Mensaje", "Etiqueta de referencia", "v2", "v8", "Fuente"], [[r["text"], r["target"], r["v2_prediction"], r["v8_prediction"], r["dataset"]] for r in changed]) + "</details>"
    body += "<p>Fuentes primarias: <a href='https://aclanthology.org/L10-1209/'>Anotación humana y revisión de DIHANA</a>, <a href='https://doi.org/10.5281/zenodo.5572470'>archivo original DIHANA</a>, <a href='https://aclanthology.org/2021.eacl-main.257/'>MTOP</a>.</p>"
    body += para(f"Ejecución {s['run_id']}. Protocolo: manifest.json. Procedencia: label-provenance.json. Prueba: summary.json y componentes JSONL. Decisión: activation.json. Verificación: verification.json.")
    page = "<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>v8 · evaluación ampliada</title><style>body{margin:0;background:#f4f7fa;color:#183041;font:17px/1.55 system-ui}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:2rem}h2{margin-top:2rem}.status{background:#dcebf2;padding:20px;border-left:5px solid #166b80;font-weight:650}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #dce5ec}th{background:#dfebf2}img{width:100%;margin-top:20px}details{margin:25px 0}summary{cursor:pointer;font-weight:650}a{color:#146580}</style><main>" + body + "</main></html>"
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    (run_dir / "report.md").write_text("# Evaluación ampliada de v8\n\n" + status + "\n\n" + opening + "\n\n" + data + "\n\n" + stats + "\n\n" + calibration + "\n\n" + scope + "\n", encoding="utf-8")
