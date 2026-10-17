"""Report the expanded general bank and the predeclared v2 activation comparison."""
import base64
from collections import Counter
import html
from pathlib import Path

from .eval_report import figures,percent
from .focus_experiment import read_json
from .config import save_json
from .general_probability_audit import audit


def render(run_dir):
    run_dir=Path(run_dir)
    s=read_json(run_dir/"summary.json")
    manifest=read_json(run_dir.parents[1]/"manifest.json")
    development=read_json(run_dir/"development-selection.json")
    selection=read_json(run_dir/"selection.json")
    activation=read_json(run_dir/"activation.json")
    cases=read_json(run_dir/"test-cases.json")
    edition=s["schema_version"]
    if edition==6:
        cases=[r|{"v5_prediction":r["v6_prediction"],"v5_correct":r["v6_correct"],"v5_confidence":r["v6_confidence"]} for r in cases]
    before,raw,after=s["baseline"]["calibrated"],s["test"]["raw"],s["test"]["calibrated"]
    gen=lambda item:next(r for r in item["per_class"] if r["label"]=="general")
    gb,ga=gen(before),gen(after)
    paired=s["paired_comparison"]
    lo,hi=paired["paired_bootstrap_95ci"]
    flo,fhi=paired["focus_f1_change_95ci"]["general"]
    if activation["activated"]:
        status="V5 activada como versión predeterminada: superó los criterios fijados y las comprobaciones del servicio."
    elif activation["approved"]:
        status="V5 supera el criterio estadístico; la activación espera las comprobaciones locales. V2 sigue predeterminada."
    else:
        status="V5 no supera todos los criterios fijados. No se activa; v2 calibrada sigue predeterminada."
    opening=(f"La exactitud en {after['n']} mensajes nuevos pasa de {percent(before['accuracy'])} (v2) a {percent(after['accuracy'])} (v5). "
             f"Cambio: {100*paired['accuracy_gain']:+.2f} puntos, IC95% pareado [{100*lo:+.2f}, {100*hi:+.2f}]. "
             f"Se corrigen {paired['fixed_errors']} errores y se introducen {paired['introduced_errors']}. "
             f"McNemar unilateral predefinido: p={activation['mcnemar_one_sided_pvalue']:.6g}.")
    general_text=(f"General: F1 {percent(gb['f1'])} → {percent(ga['f1'])}, "
                  f"cambio {100*(ga['f1']-gb['f1']):+.2f} puntos, IC95% [{100*flo:+.2f}, {100*fhi:+.2f}], con {ga['support']} casos. "
                  f"Precisión {percent(gb['precision'])} → {percent(ga['precision'])}; recall {percent(gb['recall'])} → {percent(ga['recall'])}.")
    expansion=(f"El banco crece de 10.364 a {manifest['splits']['training']['count']:,} mensajes reales etiquetados. "
               f"Los de general pasan de {manifest['general_training_before']} a {manifest['general_training_after']} "
               f"(+{100*(manifest['general_training_after']/manifest['general_training_before']-1):.1f}%). "
               "Se incorpora el resto del train limpio y validaciones anteriores fuera del desarrollo actual. "
               f"También se incorporan {manifest['retired_test_general_count']} ejemplos general de pruebas anteriores: "
               "estas pruebas se retiran explícitamente y no evalúan esta versión. No se duplican ejemplos ni se cambian etiquetas humanas.")
    method=("Se vuelve a ajustar el vectorizador TF-IDF de palabras y caracteres con el banco ampliado. En cada consulta se recuperan "
            "seis vecinos diversos, con máximo dos por categoría; si faltan ejemplos de general, se añaden sus vecinos más próximos hasta tener dos "
            "(máximo ocho ejemplos en total). Se excluyen coincidencias exactas normalizadas. Se aclara general para comentarios, temas aislados "
            "y peticiones abiertas, sin inferir herramientas no solicitadas. Gemma decide entre las 18 categorías mediante sus logits reales. "
            "Es aprendizaje con ejemplos recuperados en el contexto; los pesos QAT de Gemma no se ajustan. No se mezclan puntajes de un clasificador auxiliar.")
    if edition==6:
        rule=read_json(run_dir/"task-profile.json")
        method=("Se vuelve a ajustar TF-IDF con el banco ampliado de ejemplos; los pesos QAT se conservan. "
                f"Primero se obtienen los logits reales de v2. Sólo se revisan sus ganadores ajenos a general con confianza v2 calibrada ≤{percent(rule['threshold'])}. "
                "La segunda consulta usa seis vecinos diversos del banco ampliado y añade ejemplos de general hasta tener dos (máximo ocho). "
                f"Sólo una respuesta general de esa consulta con puntaje original ≥{percent(rule['minimum_probability'])} puede intervenir. "
                f"La probabilidad general se aumenta mediante una media con peso {percent(rule['weight'])} para la nueva consulta; "
                "las demás probabilidades conservan las proporciones originales de v2. Se conservan sus ganadores general originales. "
                "Una temperatura nueva calibra después la composición completa. El runtime usa una o dos inferencias por mensaje. "
                "La evaluación recoge cada componente por lotes y deriva explícitamente la composición; no presenta esas filas derivadas como llamadas reales a v6. "
                "La API se contrasta con la fórmula usando sus componentes reales y se mide por separado la variación de caché Vulkan.")
    protocol=("Los 450 mensajes de desarrollo son validaciones oficiales ya usadas, separados del entrenamiento ampliado. "
              "Los nuevos 250 mensajes de calibración, 150 de validación y 500 de prueba excluyen todas las muestras de evaluaciones anteriores, "
              "y no comparten IDs ni textos normalizados con entrenamiento, desarrollo ni entre sí. La nueva prueba contiene 50 general y "
              "450 proporcionales de las otras 17 categorías. General está enriquecida al 10%; las métricas globales describen esta distribución. "
              "Quedan muy pocas noticias y algunas otras clases: se muestran soportes y no se infiere una mejora fiable de clases con pocos casos. "
              "La referencia es v2 calibrada en estos mismos 500 mensajes, con su temperatura congelada anterior.")
    dev_text=(f"En desarrollo: exactitud {percent(development['baseline']['accuracy'])} → {percent(development['candidate']['accuracy'])}; "
              f"F1 general {percent(gen(development['baseline'])['f1'])} → {percent(gen(development['candidate'])['f1'])}. "
              "El criterio de desarrollo y el de activación quedaron escritos en el manifiesto antes de inferir los nuevos holdouts.")
    if edition==6:
        dev_text+=" La ampliación directa mejoró general pero perdió un acierto global; se probaron 36 reglas de rescate sólo con esos mismos 450 registros de desarrollo. La corrección seleccionada se congeló antes de abrir calibración, validación o prueba."
    calibration=(f"T candidata={s['temperature']:.6f}; T seleccionada={s['selected_temperature']:.6f}. "
                 f"NLL en la validación nueva: {selection['raw_validation']['negative_log_loss']:.4f} → "
                 f"{selection['candidate_validation']['negative_log_loss']:.4f}. "
                 f"Calibración aprobada: {'sí' if s['validation_approved'] else 'no'}. "
                 "La temperatura se ajusta sólo con los 250 mensajes de calibración y se acepta si mejora NLL de validación más de 0,0001. "
                 "Temperatura y aceptación se congelan antes de inferir el test. La temperatura escalar conserva las clases predichas. "
                 "Los resultados del test no modifican temperatura, prompt ni recuperación; sólo deciden si activar la versión.")
    condition_labels={
        "calibration_validated":"Nueva calibración validada",
        "general_f1_gain_at_least_3_points":"F1 general mejora al menos 3 puntos",
        "general_precision_not_lower":"Precisión general no inferior a v2",
        "overall_accuracy_higher":"Exactitud global superior a v2",
        "macro_f1_not_lower":"F1 macro no inferior a v2",
        "paired_one_sided_p_below_0_05":"McNemar unilateral p < 0,05",
        "test_nll_not_higher":"NLL de prueba no superior a v2",
    }
    conditions=[[condition_labels[k],"Sí" if v else "No"] for k,v in activation["conditions"].items()]
    metric_rows=[]
    for label,key,fmt in (("Exactitud","accuracy",percent),("Precisión macro","macro_precision",percent),
                          ("Recall macro","macro_recall",percent),("F1 macro","macro_f1",percent),
                          ("NLL ↓","negative_log_loss",lambda x:f"{x:.4f}"),("Brier (suma) ↓","brier_multiclass_sum",lambda x:f"{x:.4f}"),
                          ("ECE (10 intervalos) ↓","ece_10_bins",percent),("Errores con confianza ≥90%","errors_with_confidence_ge_0_9",str)):
        metric_rows.append([label]+[fmt(item[key]) for item in (before,raw,after)])
    probability_audit=audit(run_dir,s)
    save_json(run_dir/"general-probability-audit.json",probability_audit)
    general_probability_rows=[[label,fmt(probability_audit["v2"][key]),fmt(probability_audit["candidate"][key])]
        for label,key,fmt in (("Brier binario ↓","binary_brier",lambda x:f"{x:.4f}"),
                              ("NLL binaria ↓","binary_nll",lambda x:f"{x:.4f}"),
                              ("ECE de p(general) ↓","ece_10_bins",percent),
                              ("NLL sólo en general verdaderos ↓","positive_class_nll",lambda x:f"{x:.4f}"))]
    per_class=[[a["label"],str(a["support"]),percent(b["f1"]),percent(a["f1"]),percent(b["precision"]),percent(a["precision"]),percent(a["recall"])]
               for b,a in zip(before["per_class"],after["per_class"])]
    figures(run_dir,s,selected_title="Nueva calibración validada" if s["validation_approved"] else "T=1 · calibración rechazada")
    fixed=[r for r in cases if not r["v2_correct"] and r["v5_correct"]]
    broken=[r for r in cases if r["v2_correct"] and not r["v5_correct"]]
    remaining=[r for r in cases if r["target"]=="general" and not r["v5_correct"]]
    fixed.sort(key=lambda r:r["target"]!="general")
    rows=lambda values:[[r["text"],r["target"],r["v2_prediction"],r["v5_prediction"]] for r in values[:15]]
    confusion=Counter((r["target"],r["v5_prediction"]) for r in cases if not r["v5_correct"] and (r["target"]=="general" or r["v5_prediction"]=="general"))
    limits=["General es heterogénea y algunas etiquetas del corpus son ambiguas; todos los errores conservan la etiqueta original.",
            "MASSIVE es público y podría haber formado parte del preentrenamiento de Gemma. IDs/textos disjuntos no descartan contaminación ni paráfrasis relacionadas.",
            "Los intervalos asumen mensajes independientes y pueden ser optimistas por paráfrasis. 50 casos general todavía dejan incertidumbre relevante.",
            "ECE depende de sus intervalos. NLL, Brier, precisión, recall y F1 se muestran conjuntamente; una mejora no garantiza las demás.",
            "El resultado se limita a estas 18 categorías de solicitudes de asistente en español. No calibra mensajes de otra taxonomía ni imagen/audio/video.",
            "Las pruebas anteriores que aportan ejemplos de entrenamiento ya no son pruebas válidas de v5. Esta iteración consume los 50 general que quedaban sin utilizar; futuras mejoras necesitarán datos independientes adicionales."]
    verification_text="Verificación local pendiente."
    if (run_dir/"verification.json").exists():
        proof=read_json(run_dir/"verification.json")
        verification_text=f"Verificación {'correcta' if proof['passed'] else 'fallida'}; {proof['unit_tests_passed']} pruebas unitarias. API comprobada: {proof.get('live',{}).get('passed',False)}."
        if "initial_to_warm_max_probability_delta" in proof.get("live",{}):
            delta=proof["live"]["initial_to_warm_max_probability_delta"]
            limits.append(f"La repetición inicial/caché Vulkan varió hasta {100*delta:.3f} puntos de probabilidad. El orden de opciones se contrasta con el mismo contexto calentado; no se garantiza igualdad bit a bit entre estados de caché.")
    calibrated="True" if s["validation_approved"] else "False"
    usage=('python -m jev_qat predict --input examples/decision-massive-v5.json\n'
           'python scripts/verify_spanish_v5.py --live\n\n'
           'from jev_qat import classify_spanish_message\n'
           f'result = classify_spanish_message("Hola, ¿cómo estás?", version="v5", calibrated={calibrated})')
    sources=("Datos originales CC BY 4.0, FitzGerald et al. / Amazon Science; espejo MTEB Apache-2.0. "
             "Revisión y SHA-256 de origen fijados en manifiestos. "
             "[MASSIVE](https://github.com/alexa/massive), [MTEB](https://huggingface.co/datasets/mteb/amazon_massive_scenario), "
             "[temperature scaling](https://arxiv.org/abs/1706.04599), [TypeSafe: validar la incertidumbre](https://docs.typesafe.ai/confidence).")
    def md_table(headers,values):
        clean=lambda x:str(x).replace("|","\\|").replace("\n"," ")
        return "| "+" | ".join(headers)+" |\n|"+"---|"*len(headers)+"\n"+"\n".join("| "+" | ".join(map(clean,r))+" |" for r in values)
    mh=["Métrica","V2 calibrada","V5 original","V5 seleccionada"]
    ch=["Categoría","Casos","F1 v2","F1 v5","Precisión v2","Precisión v5","Recall v5"]
    eh=["Mensaje","Etiqueta humana","V2","V5"]
    sections=[("Más ejemplos reales",expansion),("Clasificación con ejemplos",method),("Datos nuevos y desarrollo",protocol+" "+dev_text),("Recalibración",calibration)]
    md="# Más ejemplos para general · evaluación v5\n\n"+status+"\n\n"+opening+"\n\n"+general_text+"\n\n"+md_table(mh,metric_rows)
    md+="".join(f"\n\n## {title}\n\n{text}" for title,text in sections)
    md+="\n\n## Condiciones de activación\n\n"+md_table(["Criterio fijado","Resultado"],conditions)+"\n\n![Fiabilidad](reliability.png)"
    md+="\n\n## Probabilidad de general frente al resto\n\n"+md_table(["Métrica descriptiva","V2","Candidata"],general_probability_rows)+"\n\nEsta auditoría no ajusta parámetros; describe la muestra enriquecida al 10% de general."
    md+="\n\n## Resultados por categoría\n\n"+md_table(ch,per_class)+"\n\n![Confusión](confusion.png)"
    for title,values in (("Errores corregidos",fixed),("Errores introducidos",broken),("General: errores restantes",remaining)):
        md+=f"\n\n## {title}\n\n"+md_table(eh,rows(values))
    md+="\n\n## Uso\n\n```text\n"+usage+"\n```\n\n## Verificación\n\n"+verification_text+"\n\n## Alcance\n\n"+"\n".join("- "+v for v in limits)
    md+=f"\n\n## Evidencia\n\nEjecución {s['run_id']}. Ver manifest.json, summary.json, activation.json, selection.json, test-cases.json, source/ y las predicciones JSONL.\n\n"+sources+"\n"
    if edition==6:
        md=md.replace("v5","v6").replace("V5","V6")
    (run_dir/"report.md").write_text(md,encoding="utf-8")
    esc=lambda x:html.escape(str(x))
    p=lambda text:"<p>"+esc(text)+"</p>"
    def table(headers,values):
        return '<div class="table"><table><thead><tr>'+"".join("<th>"+esc(v)+"</th>" for v in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+esc(v)+"</td>" for v in r)+"</tr>" for r in values)+"</tbody></table></div>"
    plot=lambda name:f'<img alt="{esc(name)}" src="data:image/png;base64,{base64.b64encode((run_dir/name).read_bytes()).decode()}">'
    body="<h1>Más ejemplos para general · evaluación v5</h1><p class=decision>"+esc(status)+"</p>"+p(opening)+p(general_text)+table(mh,metric_rows)
    body+="".join("<h2>"+esc(title)+"</h2>"+p(text) for title,text in sections)
    body+="<h2>Condiciones de activación</h2>"+table(["Criterio fijado","Resultado"],conditions)+plot("reliability.png")
    body+="<h2>Probabilidad de general frente al resto</h2>"+table(["Métrica descriptiva","V2","Candidata"],general_probability_rows)+p("Auditoría descriptiva: no ajusta parámetros y describe esta muestra enriquecida al 10% de general.")
    body+="<h2>Resultados por categoría</h2>"+table(ch,per_class)+plot("confusion.png")
    body+="<h2>Confusiones que afectan a general</h2>"+table(["Etiqueta humana","V5","Casos"],[[a,b,str(n)] for (a,b),n in confusion.most_common()])
    for title,values in (("Errores corregidos",fixed),("Errores introducidos",broken),("General: errores restantes",remaining)):
        body+="<h2>"+esc(title)+"</h2>"+table(eh,rows(values))
    body+="<h2>Uso</h2><pre>"+esc(usage)+"</pre><h2>Verificación</h2>"+p(verification_text)
    body+="<h2>Alcance</h2><ul>"+"".join("<li>"+esc(v)+"</li>" for v in limits)+"</ul>"
    body+=p(f"Ejecución {s['run_id']}. Evidencia: summary.json, activation.json, manifest.json, selection.json, test-cases.json, source/ y JSONL.")
    body+='<p>Datos CC BY 4.0, FitzGerald et al. / Amazon Science; espejo Apache-2.0. <a href="https://github.com/alexa/massive">MASSIVE</a> · <a href="https://huggingface.co/datasets/mteb/amazon_massive_scenario">MTEB</a> · <a href="https://arxiv.org/abs/1706.04599">Guo et al.</a> · <a href="https://docs.typesafe.ai/confidence">TypeSafe</a>.</p>'
    if edition==6:
        body=body.replace("v5","v6").replace("V5","V6")
    (run_dir/"report.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>General · más ejemplos · v{edition}</title><style>body{{font:16px/1.55 system-ui,sans-serif;color:#182838;max-width:1150px;margin:40px auto;padding:0 24px}}'
        'h1{font-size:32px}h2{margin-top:36px}.decision{background:#eef3f7;padding:16px;border-left:4px solid #145c94}.table{overflow:auto}'
        'table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e9}th{background:#eef3f7}'
        'img{width:100%;height:auto}pre{padding:16px;background:#eef3f7;overflow:auto}li{margin-bottom:8px}a{color:#125c9a}</style>'+body+'</html>',encoding="utf-8")
