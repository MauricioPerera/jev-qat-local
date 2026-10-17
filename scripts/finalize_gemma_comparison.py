"""Audit fresh memory measurements and publish a self-contained delivery."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "gemma-e2b-vs-e4b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    control = read(OUT / "memory-control" / "results.json")
    primary = read(OUT / "results.json")
    assert control["passed"] and control["e4b_default_healthy"] and control["e2b_healthy"]
    assert read(OUT / "verification.json")["passed"]
    for model, measurement in control["models"].items():
        samples = [json.loads(line) for line in (OUT / "memory-control" / f"samples-{model}.jsonl").read_text(encoding="utf-8").splitlines()]
        selected = []
        for trial in measurement["trials"]:
            assert trial["result"]["metadata"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            current = [s for s in samples if s["end"] > trial["start"] and s["start"] < trial["end"]]
            coverage = sum(min(s["end"],trial["end"])-max(s["start"],trial["start"]) for s in current)
            assert abs(coverage-(trial["end"]-trial["start"])) < 1e-6
            selected += current
        for metric, field in (("rss_peak_gib","rss_bytes"),("commit_peak_gib","private_committed_bytes"),("gpu_shared_peak_gib","gpu_shared_bytes")):
            assert abs(max(s[field] for s in selected)/1024**3-measurement[metric]) < 1e-9
    suites = list(ET.parse(ROOT / "inference" / "e2b-unit-tests.xml").getroot().iter("testsuite"))
    tests = sum(int(s.attrib["tests"]) for s in suites)
    assert tests == 97 and sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0)) for s in suites)==0
    models = {m:{k:v for k,v in data.items() if k not in ("trials","backend_pid")} for m,data in control["models"].items()}
    a,b = models["e4b"],models["e2b"]
    savings = {k:100*(a[k]-b[k])/a[k] for k in ("rss_peak_gib","commit_peak_gib","gpu_shared_peak_gib")}
    summary = read(OUT / "summary.json")
    summary["fresh_process_memory_control"] = {"models":models,"reduction_percent":savings,"measured_requests":8,
                                                 "warmup_requests":2,"original_e4b_stays_loaded":True,"verified":True}
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    old_resident = {m:max(r[m]["rss_peak_gib"] for r in primary["comparison"]) for m in ("e4b","e2b")}
    lines = ["## Control adicional de RAM con procesos recién arrancados", "",
             f"En la comparación principal, el pico residente observado fue {old_resident['e4b']:.2f} GiB para E4B y "
             f"{old_resident['e2b']:.2f} GiB para E2B. Los procesos tenían estados de residencia de Windows distintos: "
             "el proceso E4B antiguo conservaba gran parte de su memoria privada comprometida fuera de su conjunto residente. "
             "Se añadió un control separado con procesos nuevos para comparar la memoria desde condiciones de arranque equivalentes.", "",
             "Se mantuvo el E4B original cargado e inactivo como fondo común. Se detuvo temporalmente el E2B de prueba; "
             "se midió una instancia E4B nueva en el puerto 18168 y después una instancia E2B nueva. "
             "Cada una procesó texto v2, la foto de 45,9 MP, audio de 30 s y vídeo de 60 s/16 fotogramas. "
             "La instancia temporal E4B se cerró y el servicio E2B quedó disponible en 18167. "
             "Los tiempos de la tabla principal siguen siendo los de las 80 peticiones alternadas; este control añade ocho peticiones medidas y dos de calentamiento.", "",
             "| Modelo recién arrancado | RAM residente pico (GiB) | Privada comprometida pico (GiB) | GPU compartida pico (GiB) |",
             "|---|---:|---:|---:|"]
    for model in ("e4b","e2b"):
        d=models[model]
        lines.append(f"| {model.upper()} | {d['rss_peak_gib']:.2f} | {d['commit_peak_gib']:.2f} | {d['gpu_shared_peak_gib']:.2f} |")
    lines += ["",f"En este control E2B redujo el pico residente un {savings['rss_peak_gib']:.1f}%, "
              f"la reserva privada un {savings['commit_peak_gib']:.1f}% y la memoria gráfica compartida un {savings['gpu_shared_peak_gib']:.1f}%. "
              "Las métricas de memoria se solapan y no se suman. Muestras y comprobaciones: memory-control/results.json y memory-control/samples-*.jsonl.", "",
              f"Verificación de entrega: {tests} pruebas unitarias aprobadas; integración HTTP/Python y ciclo de parada/arranque aprobados; "
              "hashes de los cuatro GGUF y de los medios comprobados; cero tokens reutilizados; el perfil E4B predeterminado continúa en v2."]
    markdown=OUT/"report.md"
    if "## Control adicional de RAM" not in markdown.read_text(encoding="utf-8"):
        with markdown.open("a",encoding="utf-8") as stream:stream.write("\n"+"\n".join(lines)+"\n")
    note=(f"Control de memoria con procesos recién arrancados: RAM residente {a['rss_peak_gib']:.2f} GiB (E4B) / {b['rss_peak_gib']:.2f} GiB (E2B); "
          f"GPU compartida {a['gpu_shared_peak_gib']:.2f} / {b['gpu_shared_peak_gib']:.2f} GiB. "
          "La columna RAM de la sesión principal conserva las observaciones de procesos con estados de residencia de Windows distintos; el control adicional está detallado al final.")
    text=markdown.read_text(encoding="utf-8")
    if "Control de memoria con procesos recién arrancados:" not in text:
        title,rest=text.split("\n",1)
        markdown.write_text(title+"\n\n"+note+"\n"+rest,encoding="utf-8")
    document=OUT/"report.html"
    source=document.read_text(encoding="utf-8")
    marker="<h2>Control adicional de RAM"
    if marker in source:
        source=source.split(marker,1)[0]+"</html>"
    if 'id="memory-note"' not in source:
        source=source.replace("</h1>",'</h1><p id="memory-note" style="padding:16px;background:#e7f4ef;border-left:4px solid #087f8c">'+html.escape(note)+"</p>",1)
    body=[]
    table=False
    for line in lines:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not table:
                body.append("<div class='scroll'><table>")
                table=True
            body.append("<tr>"+"".join("<td>"+html.escape(c.strip())+"</td>" for c in line.strip("|").split("|"))+"</tr>")
            continue
        if table:
            body.append("</table></div>")
            table=False
        if line.startswith("## "):
            body.append("<h2>"+html.escape(line[3:])+"</h2>")
        elif line:
            body.append("<p>"+html.escape(line)+"</p>")
    document.write_text(source.replace("</html>","".join(body)+"</html>"),encoding="utf-8")
    with httpx.Client(timeout=15,trust_env=False) as client:
        for port,alias in ((18165,"gemma-e4b-qat-q4-local"),(18167,"gemma-e2b-qat-q4-local")):
            response=client.get(f"http://127.0.0.1:{port}/health");response.raise_for_status()
            assert response.json()["backend"]==alias
        for route,file in (("comparison","report.html"),("comparison/figure","comparison.png")):
            response=client.get("http://127.0.0.1:18167/"+route);response.raise_for_status()
            assert response.content==(OUT/file).read_bytes()
        assert client.get("http://127.0.0.1:18167/comparison/summary").json()==summary
        response=client.post("http://127.0.0.1:18167/predict",json={"state":"El código es 5831.","question":"¿Cuál es el código?","options":["8513","5831","1385"],"cache_prompt":False})
        response.raise_for_status()
        assert response.json()["prediction"]=="5831" and response.json()["metadata"]["calibrated"] is False
    for name in ("control_gemma_memory.py","verify_gemma_comparison.py","plot_gemma_comparison.py","finalize_gemma_comparison.py"):
        shutil.copy2(ROOT/"scripts"/name,OUT/name)
    shutil.copy2(ROOT/"inference"/"e2b-unit-tests.xml",OUT/"unit-tests.xml")
    for name in ("integration-results.json","lifecycle-results.json"):
        shutil.copy2(ROOT/"integration"/"jev-qat-e2b"/name,OUT/("e2b-"+name))
    delivery={"passed":True,"unit_tests":tests,"primary_requests":80,"fresh_memory_requests":8,
              "native_input_cache_verified_zero":True,"both_services_healthy":True,"served_artifacts_match_disk":True,
              "e4b_default_version":"v2","e2b_probabilities_calibrated":False,"fresh_memory_verified":True}
    (OUT/"delivery-verification.json").write_text(json.dumps(delivery,indent=2)+"\n",encoding="utf-8")
    files=["results.json","summary.json","report.md","report.html","comparison.csv","comparison.png","verification.json",
           "memory-control/results.json","delivery-verification.json","unit-tests.xml"]
    hashes={f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files}
    (OUT/"artifact-sha256.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"delivery":delivery,"fresh_memory":models,"reduction_percent":savings},ensure_ascii=True))


if __name__=="__main__":
    main()
