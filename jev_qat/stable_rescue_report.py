"""Auditable Spanish report: stability, calibration and a new fixed cohort."""
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .focus_experiment import read_json
from .stable_rescue_run import general


def table(headers, rows):
    tr = lambda row, tag: "<tr>" + "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in row) + "</tr>"
    return "<div class='scroll'><table><thead>" + tr(headers, "th") + "</thead><tbody>" + "".join(tr(r, "td") for r in rows) + "</tbody></table></div>"


def render(run_dir):
    run = Path(run_dir)
    s, a = read_json(run / "summary.json"), read_json(run / "activation.json")
    m, selection = read_json(run.parents[1] / "manifest.json"), read_json(run / "selection.json")
    stability, cases = read_json(run / "stability-final.json"), read_json(run / "test-cases.json")
    b, c = s["baseline"]["calibrated"], s["test"]["calibrated"]
    pct = lambda x: f"{100*x:.2f}%"
    status = ("V9 activada tras superar todos los criterios y la verificación." if a["activated"] else
              "V9 supera los criterios; pendiente de verificación, v2 sigue activa." if a["statistical_and_quality_gates_passed"] and a["verification_required"] else
              "V9 no se activa; v2 calibrada sigue predeterminada.")
    intro = (f"Prueba nueva con {c['n']} mensajes: exactitud {pct(b['accuracy'])} → {pct(c['accuracy'])}. "
        f"{s['paired_comparison']['fixed_errors']} errores corregidos y {s['paired_comparison']['introduced_errors']} introducidos. "
        f"McNemar exacto unilateral p={a['pvalue']:.6g}; umbral fijado p < {a['alpha']}.")
    method = ("El rescate v9 conserva las instrucciones, bancos y pesos QAT. Desactiva la reutilización del prefijo en los componentes "
        "de rescate y confirmación, evaluando todo el mensaje. La referencia v2 mantiene su ejecución habitual y se comparte como "
        "componente base. La dependencia de los logits del tamaño de lote con caché está documentada por llama.cpp y se reprodujo "
        "localmente con el caso de la reunión. El corte brusco de rescate 0,95 se reemplaza por una transición continua entre 0,8 y 0,9. "
        "La confirmación también entra gradualmente: 0,5–0,7 para los dominios comunes y 0,85–0,95 para conocimiento factual. "
        "La mezcla conserva los pesos máximos 0,7. No contiene excepciones para frases ni IDs concretos.")
    development = ("Se probaron dos composiciones sólo con regresiones antiguas. La primera daba demasiado peso a una confirmación "
        "incierta de atributos y perdió el caso del iPad; sus tres pasadas se conservan en development-initial. La segunda usó rampas "
        "continuas de fiabilidad. Se comprobaron en vivo ocho recuperaciones y tres correcciones previas, en el orden original, inverso "
        "y barajado, sin repetir hasta aprobar. La primera pasada final parte del servicio recién iniciado. Son comprobaciones de "
        "regresión, no ejemplos nuevos para la prueba estadística.")
    data = (f"El test nuevo tiene {m['counts']['test']['datasets'].get('MTOP',0)} solicitudes MTOP, "
        f"{m['counts']['test']['datasets'].get('Multi3NLU++',0)} actos Multi3NLU++ y "
        f"{m['counts']['test']['datasets'].get('DIHANA',0)} actos DIHANA, además de 200 preguntas factuales Mintaka. Se excluyen todos los IDs y familias de texto expuestos, "
        "incluidas variantes que sólo difieren en puntuación. Los actos de reparación conversacional conservan la etiqueta humana "
        "de repetir/no entendido; se mapean a interacción general cuando no tienen campos de otra tarea. DIHANA exige conceptos "
        "y argumentos Nil y usa un único segmento por participante dentro del test. Multi3 usa un ejemplo por grupo publicado "
        "de dominio/pliegue; esos prefijos de ID no son identificadores de conversación ni de autor. Los grupos de calibración "
        "están separados del test. Los textos e intenciones MTOP se contrastan con el archivo original de Meta. Mintaka aporta "
        "preguntas de historia, geografía y libros escritas y anotadas por personas, con traducción profesional al español de México. "
        "Se incluyen antes de inferir para comprobar el efecto del nuevo tratamiento de confirmaciones sobre conocimiento factual.")
    labels = ("Se usan únicamente etiquetas humanas originales documentadas y sus textos españoles. No se generan referencias "
        "con un modelo. El mapeo determinista entre ontologías se congela antes de inferir; no recibe una nueva adjudicación humana "
        "para nuestras 18 categorías. Se conserva toda referencia después de puntuar. Es una muestra de desafío con general "
        "conversacional, siete dominios específicos y conocimiento factual, no una estimación de toda conversación abierta, consejos o tráfico real. "
        "Algunos participantes DIHANA aparecen en evaluaciones anteriores, pero DIHANA no se usa para elegir parámetros ni para "
        "calibrar v9. En MTOP/Multi3 no hay IDs de autor que permitan excluir toda dependencia residual. Se desconoce la exposición "
        "de Gemma a estos datos durante su preentrenamiento.")
    calibration = (f"Se retiran 100 ejemplos públicos anteriores para ajustar una temperatura y otros 100 para validarla. "
        f"Temperatura estimada {s['temperature']:.6f}; seleccionada {s['selected_temperature']:.6f}. "
        f"NLL de validación {selection['raw_validation']['negative_log_loss']:.4f} → {selection['candidate_validation']['negative_log_loss']:.4f}. "
        "La selección y el código quedan congelados antes del primer mensaje del test nuevo. Escalar la temperatura no modifica "
        "la categoría ganadora. Ningún resultado de la muestra nueva se usa para ajustar la mezcla ni las probabilidades.")
    statistics = ("La prueba tiene tamaño fijo y una sola decisión final, con McNemar exacto unilateral p < 0,025 sobre los mismos "
        "mensajes y referencias. No se mezclan pruebas antiguas en el cálculo. El criterio de v9 exige conservar precisión y F1 de "
        "general, en lugar de sumar tres puntos de F1: ese requisito anterior era inalcanzable con el F1 inicial 98,22% de la última "
        "cohorte. Este cambio está declarado antes de abrir el test de v9. La superioridad estadística en exactitud, conservar las "
        "ocho recuperaciones y todas las otras comprobaciones siguen siendo obligatorias. Un resultado sin significación no "
        "demuestra equivalencia. Los intervalos bootstrap al 95% son descriptivos y pueden degenerar si no hay discordancias.")
    metrics_rows = [[name, fmt(b[key]), fmt(c[key])] for name, key, fmt in (
        ("Exactitud", "accuracy", pct), ("F1 macro (18, incluidas sin soporte)", "macro_f1", pct),
        ("NLL ↓", "negative_log_loss", lambda x:f"{x:.4f}"), ("Brier ↓", "brier_multiclass_sum", lambda x:f"{x:.4f}"),
        ("ECE ↓", "ece_10_bins", pct))]
    gr = [[name, fmt(general(b)[key]), fmt(general(c)[key])] for name, key, fmt in (
        ("Precisión", "precision", pct), ("Recall", "recall", pct), ("F1", "f1", pct), ("Soporte", "support", str))]
    ids = list(dict.fromkeys(r["id"] for r in stability["results"]))
    regressions = [[identifier, next(r["target"] for r in stability["results"] if r["id"]==identifier)] +
        ["Correcto" if next(r["passed"] for r in stability["results"] if r["id"]==identifier and r["pass"]==i) else "Fallo" for i in (1,2,3)] for identifier in ids]
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,value,title,color in zip(axes,(b,c),("v2 calibrada","v9 calibrada"),("#64748b","#0e7490")):
        rr=[r for r in value["reliability"] if r["count"]]
        ax.plot([0,1],[0,1],"--",color="#999999")
        ax.scatter([r["confidence"] for r in rr],[r["accuracy"] for r in rr],s=[30+r["count"] for r in rr],color=color)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Confianza media",ylabel="Exactitud observada",title=title)
        ax.grid(alpha=.15)
    fig.suptitle(f"Fiabilidad en datos nuevos · n={c['n']} · calibración fijada antes del test")
    fig.savefig(run/"reliability.png",dpi=160);fig.savefig(run/"reliability.svg");plt.close(fig)
    image=base64.b64encode((run/"reliability.png").read_bytes()).decode()
    para=lambda text:"<p>"+html.escape(text)+"</p>"
    body="<h1>Rescate estable v9 frente a v2</h1><div class='status'>"+html.escape(status)+"</div>"+para(intro)
    body+=table(["Métrica","v2","v9"],metrics_rows)+"<h2>General</h2>"+table(["Métrica","v2","v9"],gr)
    body+="<h2>Corrección y estabilidad</h2>"+para(method)+para(development)+table(["Caso antiguo","Referencia","Orden original","Orden inverso","Orden barajado"],regressions)
    body+="<h2>Datos y etiquetas humanas</h2>"+para(data)+para(labels)
    body+=table(["Fuente","N","Exactitud v2","Exactitud v9"],[[name,value["candidate"]["n"],pct(value["baseline"]["accuracy"]),pct(value["candidate"]["accuracy"])] for name,value in s["per_source"].items()])
    body+="<h2>Calibración y contraste</h2>"+para(calibration)+para(statistics)+table(["Criterio fijado","Cumple"],[[k,"Sí" if v else "No"] for k,v in a["conditions"].items()])
    body+=f"<img alt='Fiabilidad de las probabilidades de v2 y v9' src='data:image/png;base64,{image}'>"
    changed=[r for r in cases if r["v2_prediction"]!=r["v9_prediction"]]
    body+="<details><summary>Decisiones modificadas del test nuevo</summary>"+table(["Texto","Referencia","v2","v9","Fuente"],[[r["text"],r["target"],r["v2_prediction"],r["v9_prediction"],r["dataset"]] for r in changed])+"</details>"
    body+="<p>Fuentes: <a href='https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/README.md'>Caché y logits de llama.cpp</a>, <a href='https://aclanthology.org/2023.findings-acl.230/'>Multi3NLU++</a>, <a href='https://aclanthology.org/L10-1209/'>Anotación humana de DIHANA</a>, <a href='https://aclanthology.org/2021.eacl-main.257/'>MTOP</a>, <a href='https://aclanthology.org/2022.coling-1.138/'>Mintaka</a>.</p>"
    body+=para(f"Ejecución {s['run_id']}. Protocolo: manifest.json. Selección: selection.json. Congelación: pre-test-freeze.json. Componentes nativos: JSONL. Decisión: activation.json. Verificación: verification.json.")
    page="<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>v9 · rescate estable</title><style>body{margin:0;background:#f4f7fa;color:#183041;font:17px/1.55 system-ui}main{max-width:1120px;margin:auto;padding:32px}h1{font-size:2rem}h2{margin-top:2rem}.status{padding:20px;background:#dcebf2;border-left:5px solid #0e7490;font-weight:650}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px 14px;border-bottom:1px solid #dce5ec;text-align:left}th{background:#dfebf2}img{width:100%;margin-top:20px}details{margin:25px 0}summary{cursor:pointer;font-weight:650}a{color:#126580}</style><main>"+body+"</main></html>"
    (run/"report.html").write_text(page,encoding="utf-8")
    (run/"report.md").write_text("# Rescate estable v9\n\n"+status+"\n\n"+intro+"\n\n"+method+"\n\n"+data+"\n\n"+labels+"\n\n"+calibration+"\n\n"+statistics+"\n",encoding="utf-8")
