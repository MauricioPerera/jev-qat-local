"""Self-contained report, separating retired regressions from new provisional data."""
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import save_json
from .eval_data import CATEGORIES
from .eval_metrics import wilson_interval
from .focus_experiment import read_json
from .general_precision_run import general
from .general_probability_audit import audit


def figures(run_dir, summary):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
    for ax,key,title,color in zip(axes,("raw","calibrated"),("Puntajes originales v7","Probabilidades seleccionadas v7"),("#b05a12","#145c94")):
        rows = [r for r in summary["test"][key]["reliability"] if r["count"]]
        x = np.array([r["confidence"] for r in rows]); y = np.array([r["accuracy"] for r in rows])
        ci = np.array([wilson_interval(round(r["accuracy"]*r["count"]),r["count"]) for r in rows])
        ax.plot([0,1],[0,1],"--",color="#87919a")
        ax.errorbar(x,y,yerr=np.maximum([y-ci[:,0],ci[:,1]-y],0),fmt="none",ecolor=color,capsize=4)
        ax.scatter(x,y,s=[35+r["count"] for r in rows],color=color)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Confianza media por intervalo",ylabel="Exactitud observada",title=title)
        ax.grid(alpha=.15)
    fig.suptitle(f"Fiabilidad · muestra nueva en español · n={summary['test']['raw']['n']}\nIC de Wilson al 95%; área proporcional al soporte; etiquetas OASST provisionales")
    for ext in ("png","svg"):
        fig.savefig(run_dir/f"reliability.{ext}",dpi=170)
    plt.close(fig)
    matrix = np.array(summary["test"]["raw"]["confusion_matrix"])
    normalized = np.divide(matrix,matrix.sum(axis=1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(axis=1,keepdims=True)!=0)
    fig,ax = plt.subplots(figsize=(13,11),layout="constrained")
    im = ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
    ax.set(xticks=range(18),yticks=range(18),xticklabels=list(CATEGORIES),yticklabels=list(CATEGORIES),
           xlabel="Categoría predicha",ylabel="Categoría de referencia",title="Muestra nueva MASSIVE + OASST\nColor normalizado por fila; etiquetas OASST provisionales")
    plt.setp(ax.get_xticklabels(),rotation=55,ha="right")
    for row,col in zip(*np.nonzero(matrix)):
        ax.text(col,row,str(matrix[row,col]),ha="center",va="center",fontsize=9,color="white" if normalized[row,col]>.5 else "#152538")
    fig.colorbar(im,ax=ax,shrink=.8,label="Proporción por etiqueta de referencia")
    for ext in ("png","svg"):
        fig.savefig(run_dir/f"confusion.{ext}",dpi=170)
    plt.close(fig)


def render(run_dir):
    run_dir = Path(run_dir)
    summary = read_json(run_dir/"summary.json")
    manifest = read_json(run_dir.parents[1]/"manifest.json")
    selection = read_json(run_dir/"selection.json")
    development = read_json(run_dir/"development-selection.json")
    rule = read_json(run_dir/"task-profile.json")
    cases = read_json(run_dir/"test-cases.json")
    activation = read_json(run_dir/"activation.json")
    before,raw,after = summary["baseline"]["calibrated"],summary["test"]["raw"],summary["test"]["calibrated"]
    chosen = next(t for t in development["trials"] if t["minimum_general_similarity"] == development["selected"])
    previous_run = Path(development["source_v6_run"])
    previous = read_json(previous_run/"summary.json")
    old_base,old_v6,old_v7 = general(previous["baseline"]["calibrated"]),general(previous["test"]["calibrated"]),general(chosen["details"]["test"]["metrics"])
    paired = summary["paired_comparison"]
    percent = lambda x:f"{100*x:.2f}%"
    def fp(m):
        matrix = np.array(m["confusion_matrix"]);gi=list(CATEGORIES).index("general")
        return int(matrix[:,gi].sum()-matrix[gi,gi])
    status = ("V7 activada tras superar los criterios y la verificación local." if activation["activated"] else
              "V7 supera los criterios estadísticos; pendiente de verificación local. V2 continúa predeterminada." if activation["approved"] else
              "V7 no supera todos los criterios fijados. V2 calibrada continúa predeterminada.")
    opening = (f"Prueba nueva de {after['n']} mensajes: exactitud {percent(before['accuracy'])} → {percent(after['accuracy'])}. "
               f"Se corrigen {paired['fixed_errors']} errores y se introducen {paired['introduced_errors']}. "
               f"McNemar unilateral: p={activation['mcnemar_one_sided_pvalue']:.6g}.")
    data_text = (f"La prueba contiene {manifest['splits']['test']['datasets']['MASSIVE']} mensajes MASSIVE nunca usados "
                 f"y {manifest['splits']['test']['datasets']['OpenAssistant/oasst1']} mensajes humanos de OpenAssistant en español. "
                 f"Incluye {general(after)['support']} referencias de general. No quedaban ejemplos general nuevos en el test original de MASSIVE. "
                 "Las etiquetas de intención de OpenAssistant fueron asignadas por Codex antes de inferir; son provisionales y no tienen adjudicación humana independiente. "
                 "Este ensayo mide una muestra de desafío con otra distribución, y no equivale a una prueba con etiquetas humanas independientes para general. "
                 "La independencia se comprueba respecto a los ejemplos del adaptador; no se conoce la posible presencia de estos datos públicos en el preentrenamiento de Gemma. "
                 "Se excluyeron 15 mensajes ambiguos antes de inferir; se preservan sus anotaciones y motivos. Los splits no comparten textos normalizados con entrenamiento ni evaluaciones anteriores.")
    method = (f"Se mantiene el rescate de v6 y se añade un respaldo mínimo de entrenamiento de {rule['minimum_general_similarity']:.2f} "
              "cuando v2 propone una acción sobre dispositivos o canales. Se calcula con los vecinos general del banco congelado, excluyendo el texto exacto. "
              "Una consulta ampliada muy confiada ya no basta para reclasificar una petición contextual con respaldo insuficiente. "
              "Las decisiones general originales de v2 se conservan. Los bancos y los pesos QAT permanecen iguales; ninguna etiqueta nueva se usa como ejemplo de entrenamiento. "
              "La evaluación deriva v7 de logits reales v2/v5 recogidos por lotes y conserva cada componente. Los tiempos derivados no son medidas de latencia secuencial en producción.")
    regression = (f"La prueba anterior de 500 mensajes se retira a desarrollo: no cuenta como datos nuevos. "
                  f"En ella, precisión general {percent(old_v6['precision'])} → {percent(old_v7['precision'])}; "
                  f"recall {percent(old_v6['recall'])} → {percent(old_v7['recall'])}. Los falsos positivos bajan de 7 a 6. "
                  "Se conserva la corrección de consejos de salud y del comentario sobre una reunión pasada, y se evita desviar la petición de iluminación. "
                  "También se conservan los seis rescates correctos de v6 en el desarrollo antiguo. Los seis falsos positivos heredados de v2 siguen presentes: "
                  "varios son fragmentos contextuales o tienen vecinos de entrenamiento con etiquetas contradictorias; este ajuste no inventa su contexto.")
    calibration = (f"Temperatura candidata T={summary['temperature']:.6f}; seleccionada T={summary['selected_temperature']:.6f}. "
                   f"Se ajustó con {summary['sample_counts']['calibration']} mensajes nuevos y se validó con otros {summary['sample_counts']['validation']}. "
                   f"NLL de validación {selection['raw_validation']['negative_log_loss']:.4f} → {selection['candidate_validation']['negative_log_loss']:.4f}. "
                   f"Validación {'aprobada' if summary['validation_approved'] else 'rechazada'}. Temperatura y aceptación se congelaron antes del test. "
                   "La calibración describe la distribución de esta muestra; la temperatura no modifica las clases predichas.")
    new_error_path = run_dir/"new-test-error-analysis.json"
    new_error_text = ("Los tres nuevos falsos positivos general tienen etiquetas MASSIVE originales: una pregunta factual sobre Mario Casas, "
                      "un topónimo y una petición de sonidos. V2 acertaba la primera; ya fallaba las otras dos en otros dominios. "
                      "El nuevo acierto general viene de OASST, con referencia provisional. El respaldo de entrenamiento para herramientas "
                      "no basta para impedir rescates erróneos desde qa, datetime o music. Estos resultados sólo determinan rechazar v7; "
                      "no se modificaron parámetros después del test." if new_error_path.exists() else "")
    metrics_rows = [[label,fmt(before[key]),fmt(raw[key]),fmt(after[key])] for label,key,fmt in
                    (("Exactitud","accuracy",percent),("F1 macro","macro_f1",percent),
                     ("NLL ↓","negative_log_loss",lambda x:f"{x:.4f}"),("Brier (suma) ↓","brier_multiclass_sum",lambda x:f"{x:.4f}"),
                     ("ECE (10 intervalos) ↓","ece_10_bins",percent))]
    general_rows = [[label,fmt(general(before)[key]),fmt(general(after)[key])] for label,key,fmt in
                    (("Precisión","precision",percent),("Recall","recall",percent),("F1","f1",percent),("Soporte","support",str))]
    general_rows.append(["Falsos positivos",str(fp(before)),str(fp(after))])
    probability_audit = audit(run_dir,summary)
    save_json(run_dir/"general-probability-audit.json",probability_audit)
    probability_rows = [[label,f"{probability_audit['v2'][key]:.4f}",f"{probability_audit['candidate'][key]:.4f}"] for label,key in
                        (("Brier general/resto ↓","binary_brier"),("NLL general/resto ↓","binary_nll"),("ECE general/resto ↓","ece_10_bins"))]
    source_rows = [[name,str(item["candidate"]["n"]),percent(item["baseline"]["accuracy"]),percent(item["candidate"]["accuracy"]),
                    str(fp(item["baseline"])),str(fp(item["candidate"]))] for name,item in summary["per_source"].items()]
    class_rows = [[r["label"],str(r["support"]),percent(r["precision"]),percent(r["recall"]),percent(r["f1"])] for r in after["per_class"]]
    changed = [r for r in cases if r["v2_prediction"] != r["v7_prediction"]]
    general_errors = [r for r in cases if r["v7_prediction"] == "general" and r["target"] != "general"]
    error_rows = [[r["text"],r["target"],r["v2_prediction"],r["v7_prediction"],r["dataset"]] for r in changed+general_errors if r in changed or r not in changed]
    # Avoid displaying a changed false positive twice.
    error_rows = list(dict.fromkeys(tuple(r) for r in error_rows))
    gates = [[k,"Sí" if v else "No"] for k,v in activation["conditions"].items()]
    analysis_path = run_dir/"remaining-error-analysis.json"
    old_error_rows = ([[r["text"],r["target"],"Nuevo rescate" if r["introduced_by_rescue"] else "Heredado de v2",r["analysis"]]
                       for r in read_json(analysis_path)["cases"]] if analysis_path.exists() else [])
    def table(headers,rows):
        return "<div class='scroll'><table><thead><tr>"+"".join("<th>"+html.escape(h)+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join("<td>"+html.escape(str(c))+"</td>" for c in r)+"</tr>" for r in rows)+"</tbody></table></div>"
    def para(text):
        return "<p>"+html.escape(text)+"</p>"
    figures(run_dir,summary)
    images = "".join(f"<img alt='{name}' src='data:image/png;base64,{base64.b64encode((run_dir/(name+'.png')).read_bytes()).decode()}'>" for name in ("reliability","confusion"))
    body = ("<h1>Rescate de general · v7 frente a v2</h1><div class='status'>"+html.escape(status)+"</div>"+para(opening)+
            "<h2>Resultados con datos nuevos</h2>"+table(["Métrica","v2 calibrada","v7 original","v7 seleccionada"],metrics_rows)+
            "<h2>General</h2>"+table(["Métrica","v2","v7"],general_rows)+para(new_error_text)+para(data_text)+
            "<h2>Por fuente</h2>"+table(["Fuente","N","Exactitud v2","Exactitud v7","FP general v2","FP general v7"],source_rows)+
            "<h2>Ajuste y regresiones conocidas</h2>"+para(method)+para(regression)+
            "<details><summary>Análisis de los siete falsos positivos anteriores</summary>"+
            table(["Mensaje","Referencia","Origen","Análisis"],old_error_rows)+"</details>"+
            "<h2>Recalibración</h2>"+para(calibration)+table(["Auditoría descriptiva","v2","v7"],probability_rows)+images+
            "<h2>Decisión de activación</h2>"+table(["Criterio congelado","Cumple"],gates)+
            "<details><summary>Errores general y decisiones modificadas en la prueba nueva</summary>"+
            table(["Mensaje","Referencia","v2","v7","Fuente"],error_rows)+"</details>"+
            "<details><summary>Las 18 categorías</summary>"+table(["Clase","Soporte","Precisión","Recall","F1"],class_rows)+"</details>"+
            "<p>Fuentes: <a href='https://huggingface.co/datasets/mteb/amazon_massive_scenario'>MASSIVE</a> · "
            "<a href='https://huggingface.co/datasets/OpenAssistant/oasst1'>OpenAssistant/oasst1</a> (Apache 2.0). "
            f"Revisión OASST {manifest['oasst_revision']}. Ejecución {summary['run_id']}. "
            "Evidencia local: manifest.json, annotations.json, development-selection.json, selection.json, pre-test-freeze.json, "
            "summary.json, activation.json y componentes JSONL. No se ajustó el rescate con este test.</p>")
    page = "<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>General v7 · evaluación</title><style>body{font:17px/1.55 system-ui;color:#192d3c;background:#f4f7fa;margin:0}main{max-width:1150px;margin:auto;padding:32px}h1{font-size:2rem}h2{margin-top:2rem}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #dce4eb}th{background:#e2edf5}td:first-child{max-width:540px}.scroll{overflow:auto}.status{background:#deebf6;border-left:5px solid #386e9d;padding:18px;font-weight:650}img{max-width:100%;background:white;margin-top:24px}details{margin:22px 0}summary{cursor:pointer;font-weight:650}a{color:#155b95}</style><main>"+body+"</main></html>"
    (run_dir/"report.html").write_text(page,encoding="utf-8")
    def mdtable(headers,rows):
        return "|"+"|".join(headers)+"|\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("|"+"|".join(str(x).replace("|","/").replace("\n"," ") for x in row)+"|" for row in rows)
    (run_dir/"report.md").write_text("# Rescate de general v7\n\n"+status+"\n\n"+opening+"\n\n"+
            mdtable(["Métrica","v2 calibrada","v7 original","v7 seleccionada"],metrics_rows)+"\n\n"+
            mdtable(["General","v2","v7"],general_rows)+"\n\n"+data_text+"\n\n"+method+"\n\n"+regression+"\n\n"+calibration+"\n\n"+
            mdtable(["Criterio","Cumple"],gates)+"\n",encoding="utf-8")
    save_json(run_dir/"report-audit.json",{"old_test_is_retired":True,"oasst_labels_provisional":True,"new_test_count":after["n"],
              "general_false_positives":{"v2":fp(before),"v7":fp(after)},"old_regression_general_precision":{"v2":old_base["precision"],"v6":old_v6["precision"],"v7":old_v7["precision"]}})
