"""Final v8 report with verified original labels and frozen crosswalk."""
import base64
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import save_json
from .eval_data import CATEGORIES
from .eval_metrics import wilson_interval
from .focus_experiment import read_json
from .general_confirm_run import general
from .general_probability_audit import audit

def figures(run_dir, summary):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
    for ax,key,title,color in zip(axes,("raw","calibrated"),("Puntajes originales v8","Probabilidades seleccionadas v8"),("#b05a12","#145c94")):
        rows = [r for r in summary["test"][key]["reliability"] if r["count"]]
        x = np.array([r["confidence"] for r in rows]); y = np.array([r["accuracy"] for r in rows])
        ci = np.array([wilson_interval(round(r["accuracy"]*r["count"]),r["count"]) for r in rows])
        ax.plot([0,1],[0,1],"--",color="#87919a")
        ax.errorbar(x,y,yerr=np.maximum([y-ci[:,0],ci[:,1]-y],0),fmt="none",ecolor=color,capsize=4)
        ax.scatter(x,y,s=[35+r["count"] for r in rows],color=color)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="Confianza media por intervalo",ylabel="Exactitud observada",title=title)
        ax.grid(alpha=.15)
    fig.suptitle(f"Fiabilidad · muestra nueva en español · n={summary['test']['raw']['n']}\nIC de Wilson al 95%; área proporcional al soporte; etiquetas originales; mapeo acotado")
    for ext in ("png","svg"):
        fig.savefig(run_dir/f"reliability.{ext}",dpi=170)
    plt.close(fig)

    matrix = np.array(summary["test"]["raw"]["confusion_matrix"])
    normalized = np.divide(matrix,matrix.sum(axis=1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(axis=1,keepdims=True)!=0)
    fig,ax = plt.subplots(figsize=(13,11),layout="constrained")
    im = ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
    ax.set(xticks=range(18),yticks=range(18),xticklabels=list(CATEGORIES),yticklabels=list(CATEGORIES),
           xlabel="Categoría predicha",ylabel="Categoría de referencia",title="Muestra nueva MASSIVE + MTOP + Multi3NLU++\nColor normalizado por fila; etiquetas originales; mapeo acotado")
    plt.setp(ax.get_xticklabels(),rotation=55,ha="right")
    for row,col in zip(*np.nonzero(matrix)):
        ax.text(col,row,str(matrix[row,col]),ha="center",va="center",fontsize=9,color="white" if normalized[row,col]>.5 else "#152538")
    fig.colorbar(im,ax=ax,shrink=.8,label="Proporción por etiqueta de referencia")
    for ext in ("png","svg"):
        fig.savefig(run_dir/f"confusion.{ext}",dpi=170)
    plt.close(fig)



def render(run_dir):
    run_dir = Path(run_dir)
    s = read_json(run_dir/"summary.json")
    manifest = read_json(run_dir.parents[1]/"manifest.json")
    selection = read_json(run_dir/"selection.json")
    activation = read_json(run_dir/"activation.json")
    development = read_json(run_dir/"development-selection.json")
    rule = read_json(run_dir/"task-profile.json")
    cases = read_json(run_dir/"test-cases.json")
    before,raw,after = s["baseline"]["calibrated"],s["test"]["raw"],s["test"]["calibrated"]
    pct = lambda v:f"{100*v:.2f}%"
    def false_positives(m):
        matrix = np.array(m["confusion_matrix"]);i = list(CATEGORIES).index("general")
        return int(matrix[:,i].sum()-matrix[i,i])
    status = ("V8 activada: superó la comparación congelada y las comprobaciones locales." if activation["activated"] else
              "V8 supera los criterios estadísticos; v2 continúa activa hasta completar las comprobaciones locales." if activation["approved"] else
              "V8 no supera todos los criterios fijados. V2 calibrada continúa activa.")
    opening = (f"Prueba nueva con {after['n']} mensajes y {general(after)['support']} casos general. "
               f"Exactitud {pct(before['accuracy'])} → {pct(after['accuracy'])}; "
               f"{s['paired_comparison']['fixed_errors']} errores corregidos y {s['paired_comparison']['introduced_errors']} introducidos. "
               f"McNemar unilateral p={activation['mcnemar_one_sided_pvalue']:.6g}.")
    method = ("V8 confirma cada decisión general propuesta por v7, incluidas las decisiones general originales de v2. "
              "La consulta adicional vuelve a elegir entre las 18 categorías con contrastes explícitos sobre hechos externos, comentarios personales, "
              "peticiones de contenido y fragmentos de destino. Usa el banco de entrenamiento congelado; no ve las predicciones anteriores ni etiquetas de la consulta. "
              f"Su distribución recibe peso {pct(rule['weight'])} sólo si el máximo de la consulta supera "
              f"{pct(rule['minimum_confirmation_probability']['qa'])} para qa o {pct(rule['minimum_confirmation_probability']['default'])} para los demás dominios. "
              "Estos son umbrales de puntajes originales seleccionados en desarrollo, no probabilidades calibradas. "
              "Se calibra después la distribución final. Los pesos QAT y los bancos v2/v5 permanecen iguales. Puede requerir de una a tres inferencias por mensaje.")
    data = (f"Calibración: {s['sample_counts']['calibration']}; validación: {s['sample_counts']['validation']}; test: {after['n']}. "
            "Los positivos general provienen de actos conversacionales puros de Multi³NLU++: afirmación, negación, saludo, despedida, "
            "agradecimiento, reconocimiento y desconocimiento, sin otra intención ni campos de tarea. Se excluye repetir por su ambigüedad con audio. "
            "MTOP aporta solicitudes especializadas con etiquetas contrastadas individualmente contra el archivo oficial de Meta. "
            f"Se añaden {manifest['new_massive_count']} mensajes MASSIVE restantes, con etiquetas originales. "
            "IDs, textos normalizados y variantes que sólo difieren en puntuación se excluyen entre conjuntos y frente a todo el entrenamiento y las evaluaciones anteriores.")
    labels = ("Se conservaron etiquetas humanas originales y traducciones profesionales documentadas; no se generaron etiquetas de referencia con un modelo. "
              "La equivalencia acotada entre las ontologías y nuestras 18 categorías se registró antes de inferir en label-crosswalk.json. "
              "No se realizó una nueva doble anotación humana de esa equivalencia. General está enriquecida con actos conversacionales: "
              "la prueba no cubre toda la variedad de consejos o creación de contenido. Los ocho casos recuperados se comprueban por separado. "
              "La procedencia incluye conversaciones reales y ejemplos escritos por humanos, traducidos al español; no equivale a tráfico nativo de producción. "
              "No se conoce si estos datos públicos formaron parte del preentrenamiento de Gemma.")
    old = ("Se retiraron las pruebas anteriores a desarrollo: 1.132 mensajes MASSIVE con etiquetas originales, sin usar las etiquetas provisionales de OASST. "
           "Se recogieron 63 confirmaciones reales, se descartaron 12 composiciones de umbral uniforme y se probaron 60 con umbrales asimétricos. La elegida conserva los ocho aciertos recuperados y corrige los tres falsos positivos recientes "
           "a sus categorías de referencia. Es evidencia de regresión, no una estimación nueva de precisión. Los parámetros quedaron congelados antes de inferir los nuevos conjuntos.")
    cal = (f"Temperatura candidata T={s['temperature']:.6f}; seleccionada T={s['selected_temperature']:.6f}. "
           f"NLL de validación {selection['raw_validation']['negative_log_loss']:.4f} → {selection['candidate_validation']['negative_log_loss']:.4f}. "
           f"Validación {'aprobada' if s['validation_approved'] else 'rechazada'}. Se ajusta con calibración y se acepta sólo si mejora NLL de validación más de 0,0001; "
           "la decisión se congela antes del test. Una temperatura escalar cambia las probabilidades, manteniendo las categorías predichas.")
    metric_rows = [[title,fmt(before[key]),fmt(raw[key]),fmt(after[key])] for title,key,fmt in
        (("Exactitud","accuracy",pct),("F1 macro","macro_f1",pct),("NLL ↓","negative_log_loss",lambda x:f"{x:.4f}"),
         ("Brier (suma) ↓","brier_multiclass_sum",lambda x:f"{x:.4f}"),("ECE (10 intervalos) ↓","ece_10_bins",pct))]
    general_rows = [[title,fmt(general(before)[key]),fmt(general(after)[key])] for title,key,fmt in
                   (("Precisión","precision",pct),("Recall","recall",pct),("F1","f1",pct),("Soporte","support",str))]
    general_rows.append(["Falsos positivos",str(false_positives(before)),str(false_positives(after))])
    source_rows = [[name,str(v["candidate"]["n"]),pct(v["baseline"]["accuracy"]),pct(v["candidate"]["accuracy"]),
                    str(false_positives(v["baseline"])),str(false_positives(v["candidate"]))] for name,v in s["per_source"].items()]
    changed = [r for r in cases if r["v2_prediction"] != r["v8_prediction"]]
    errors = [r for r in cases if r["v8_prediction"] == "general" and r["target"] != "general"]
    rows = list(dict.fromkeys((r["text"],r["target"],r["v2_prediction"],r["v8_prediction"],r["dataset"]) for r in changed+errors))
    gates = [[k,"Sí" if v else "No"] for k,v in activation["conditions"].items()]
    from .general_confirm_math import combine
    directory = run_dir.parents[1]
    development_records = {r["id"]:r for r in (json.loads(line) for line in (directory/"development.jsonl").read_text(encoding="utf-8").splitlines())}
    original_scores = read_json(directory/"development-source-scores.json")
    proposed_by_id = dict(zip(original_scores["ids"],original_scores["v7_proposed"]))
    confirmations = {r["id"]:r for r in (json.loads(line) for line in (directory/"development-confirmation.jsonl").read_text(encoding="utf-8").splitlines())}
    regression_rows = []
    for identifier in development["protected_ids"]+development["required_false_positive_ids"]:
        r = development_records[identifier];p = proposed_by_id[identifier]
        c = confirmations[identifier]["probabilities"]
        values = combine(p,c,rule["weight"],rule["minimum_confirmation_probability"])
        regression_rows.append(["Acierto protegido" if identifier in development["protected_ids"] else "Falso positivo corregido",
                                r["state"],r["target"],list(CATEGORIES)[int(np.argmax(p))],list(CATEGORIES)[int(np.argmax(values))]])
    probability = audit(run_dir,s)
    save_json(run_dir/"general-probability-audit.json",probability)
    probability_rows = [[title,f"{probability['v2'][key]:.4f}",f"{probability['candidate'][key]:.4f}"] for title,key in
                        (("Brier general/resto ↓","binary_brier"),("NLL general/resto ↓","binary_nll"),("ECE general/resto ↓","ece_10_bins"))]
    def table(headers,rows):
        return "<div class='scroll'><table><thead><tr>"+"".join("<th>"+html.escape(h)+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join("<td>"+html.escape(str(c))+"</td>" for c in r)+"</tr>" for r in rows)+"</tbody></table></div>"
    def para(text):
        return "<p>"+html.escape(text)+"</p>"
    figures(run_dir,s)
    images = "".join(f"<img alt='{name}' src='data:image/png;base64,{base64.b64encode((run_dir/(name+'.png')).read_bytes()).decode()}'>" for name in ("reliability","confusion"))
    body = ("<h1>Confirmación de general · v8 frente a v2</h1><div class='status'>"+html.escape(status)+"</div>"+para(opening)+
            table(["Métrica","v2 calibrada","v8 original","v8 seleccionada"],metric_rows)+"<h2>General</h2>"+table(["Métrica","v2","v8"],general_rows)+
            "<h2>Datos nuevos y etiquetas contrastadas</h2>"+para(data)+para(labels)+
            table(["Fuente","N","Exactitud v2","Exactitud v8","FP general v2","FP general v8"],source_rows)+
            "<h2>Ajuste y ocho regresiones obligatorias</h2>"+para(method)+para(old)+
            table(["Caso","Mensaje antiguo","Referencia","v7","v8"],regression_rows)+
            "<h2>Recalibración</h2>"+para(cal)+table(["Auditoría descriptiva del test","v2 calibrada","v8 seleccionada"],probability_rows)+images+
            "<h2>Activación</h2>"+table(["Criterio congelado","Cumple"],gates)+
            "<details><summary>Decisiones modificadas y falsos positivos general en el test nuevo</summary>"+
            table(["Mensaje","Referencia","v2","v8","Fuente"],rows)+"</details>"+
            "<p>Fuentes primarias: <a href='https://arxiv.org/html/2212.10455'>Multi³NLU++</a>, "
            "<a href='https://aclanthology.org/2021.eacl-main.257/'>MTOP</a>, "
            "<a href='https://huggingface.co/datasets/mteb/amazon_massive_scenario'>MASSIVE</a>. "
            f"Ejecución {s['run_id']}. Archivos de auditoría: manifest.json, label-crosswalk.json, development-selection.json, "
            "selection.json, pre-test-freeze.json, summary.json, activation.json y componentes JSONL. "
            "Los resultados v8 se derivan explícitamente de logits nativos recogidos por lotes; sus tiempos no son mediciones de latencia secuencial.</p>")
    page = "<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>General v8 · evaluación</title><style>body{font:17px/1.55 system-ui;color:#192d3c;background:#f4f7fa;margin:0}main{max-width:1150px;margin:auto;padding:32px}h1{font-size:2rem}h2{margin-top:2rem}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #dce4eb}th{background:#e2edf5}td:first-child{max-width:540px}.scroll{overflow:auto}.status{background:#deebf6;border-left:5px solid #386e9d;padding:18px;font-weight:650}img{max-width:100%;background:white;margin-top:24px}details{margin:22px 0}summary{cursor:pointer;font-weight:650}a{color:#155b95}</style><main>"+body+"</main></html>"
    (run_dir/"report.html").write_text(page,encoding="utf-8")
    (run_dir/"report.md").write_text("# Confirmación de general v8\n\n"+status+"\n\n"+opening+"\n\n"+data+"\n\n"+labels+"\n\n"+method+"\n\n"+old+"\n\n"+cal+"\n",encoding="utf-8")
    save_json(run_dir/"report-audit.json",{"verified_original_labels":True,"crosswalk_has_new_human_adjudication":False,"old_tests_retired":True,
              "general_false_positives":{"v2":false_positives(before),"v8":false_positives(after)},"new_test_count":after["n"]})
