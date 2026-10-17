"""Combine the HTTP cache benchmark with the isolated no-cache control."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--uncached", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cached, uncached = read(args.cached / "results.json"), read(args.uncached / "results.json")
    assert cached.get("mode", "api_cached") == "api_cached" and uncached["mode"] == "adapter_no_cache"
    assert all(cached["checks"].values()) and all(uncached["checks"].values())
    assert cached["backend_pid"] == uncached["backend_pid"]
    assert cached["fixtures"] == uncached["fixtures"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cached_rows = {r["case"]: r for r in cached["comparison"]}
    rows = []
    for r in uncached["comparison"]:
        c = cached_rows[r["case"]]
        rows.append({"case": r["case"], "label": r["label"],
                     "no_cache_seconds": r["warm_median_seconds"],
                     "cached_http_seconds": c["warm_median_seconds"],
                     "no_cache_cpu_mean_percent": r["cpu_mean_percent"],
                     "no_cache_cpu_peak_percent": r["cpu_peak_percent"],
                     "no_cache_gpu_mean_percent": r["gpu_mean_percent"],
                     "no_cache_gpu_peak_percent": r["gpu_peak_percent"],
                     "no_cache_rss_peak_gib": r["rss_peak_gib"],
                     "no_cache_committed_peak_gib": r["committed_peak_gib"],
                     "gpu_shared_peak_gib": r["gpu_shared_peak_gib"],
                     "http_cpu_mean_percent": c["cpu_mean_percent"],
                     "http_gpu_mean_percent": c["gpu_mean_percent"],
                     "http_gpu_peak_percent": c["gpu_peak_percent"],
                     "http_rss_peak_gib": c["rss_peak_gib"],
                     "http_committed_peak_gib": c["committed_peak_gib"]})
    report = {"cached_directory": str(args.cached.resolve()), "uncached_directory": str(args.uncached.resolve()),
              "total_measured_requests": len(cached["trials"]) + len(uncached["trials"]),
              "correct_measured_requests": sum(t["correct"] for t in cached["trials"] + uncached["trials"]),
              "comparison": rows, "previous_text_seconds": cached["previous_text_seconds"],
              "sample_intervals": {"http": cached["actual_sample_interval_seconds"], "no_cache": uncached["actual_sample_interval_seconds"]},
              "all_checks_passed": True,
              "sources": {str(d.resolve()): hashlib.sha256((d / "results.json").read_bytes()).hexdigest() for d in (args.cached, args.uncached)}}
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Consumo multimodal de Gemma E4B QAT Q4", "",
            "Intel Core Ultra 9 285H, 16 procesadores lógicos, 64 GB RAM, Intel Arc 140T integrada, Vulkan. "
            "Mismo backend llama.cpp b11669, contexto 4096, 8 hilos. 48 peticiones medidas: todas correctas. "
            "Cada condición tiene cuatro pasadas por caso. Los tiempos son medianas de las tres repeticiones posteriores a la primera; "
            "CPU media corresponde a esas tres repeticiones y los picos abarcan las cuatro.", "",
            "## Coste sin reutilizar la entrada", "",
            "Control aislado a través del mismo adaptador Python y backend, con cache_prompt=false en sus peticiones. "
            "Los contadores del backend confirman cero tokens de entrada reutilizados en las 24 peticiones. "
            "El cliente de inferencia es un proceso separado del monitor. El puente HTTP habitual permanece cargado e inactivo, "
            "y queda excluido de los contadores de procesos de este control. Incluye preprocesamiento de medios y transporte local por tubería.", "",
            "| Entrada | Sin caché (s) | CPU media / pico (%) | GPU media / pico (%) | RAM residente pico (GiB) | Comprometida pico (GiB) | GPU compartida pico (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['label']} | {r['no_cache_seconds']:.3f} | {r['no_cache_cpu_mean_percent']:.1f} / {r['no_cache_cpu_peak_percent']:.1f} | "
                    f"{r['no_cache_gpu_mean_percent']:.1f} / {r['no_cache_gpu_peak_percent']:.1f} | {r['no_cache_rss_peak_gib']:.2f} | "
                    f"{r['no_cache_committed_peak_gib']:.2f} | {r['gpu_shared_peak_gib']:.2f} |")
    text += ["", "## Servicio HTTP habitual con entradas repetidas", "",
             "Mismo servicio que usa la aplicación. Caché permitida y efectivamente reutilizada: el texto v2 reutiliza 740 de 741 tokens; "
             "imagen 182 de 187; audio 153 de 158; vídeo de dos fotogramas 278 de 283; vídeo de 16 fotogramas 1766 de 1767. "
             "Son casos de reutilización de la misma entrada, no estimaciones para archivos nuevos.", "",
             "| Entrada | Repetida (s) | CPU media (%) | GPU media / pico (%) | RAM residente pico (GiB) | Comprometida pico (GiB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['label']} | {r['cached_http_seconds']:.3f} | {r['http_cpu_mean_percent']:.1f} | "
                    f"{r['http_gpu_mean_percent']:.1f} / {r['http_gpu_peak_percent']:.1f} | "
                    f"{r['http_rss_peak_gib']:.2f} | {r['http_committed_peak_gib']:.2f} |")
    text += ["", "## Comparación con la medición anterior", "",
             "La petición de texto anterior tardó 1,524 s y reutilizó 573 de 741 tokens. "
             "Ahora la misma petición tarda aproximadamente 0,279 s con casi toda la entrada reutilizada; "
             "el control sin reutilización figura en la primera tabla. La diferencia se debe a condiciones distintas de caché, "
             "carga del equipo y rutas de acceso; no demuestra una mejora del modelo.", "",
             "## Alcance y lectura de los contadores", "",
             "CPU se normaliza sobre los 16 procesadores lógicos. La media utiliza tiempos de CPU de los dos procesos principales; "
             "el pico incluye los procesos hijos observados. Los picos de GPU usan el motor más ocupado atribuible a esos procesos, "
             "no la suma de motores. Se solicitó muestreo cada 100 ms; el intervalo real está guardado en comparison.json. "
             "Procesos o picos más breves pueden no observarse. Otros programas estuvieron usando CPU durante la prueba; "
             "no es una medición en una máquina desocupada.", "",
             "RAM residente es la suma de conjuntos de trabajo de los procesos observados. Memoria comprometida es memoria privada virtual "
             "y no equivale a RAM física ocupada. La GPU utiliza RAM compartida: estas tres cifras se solapan y no se deben sumar. "
             "Los buffers quedan asignados después de solicitudes grandes, por lo que los picos por modalidad dependen del orden y "
             "no representan incrementos independientes. El cliente Python del control y el puente HTTP tienen distintas reservas de memoria.", "",
             "La imagen mide 640×360; el audio dura 3,297 s, mono a 22050 Hz y se convierte a 16 kHz; "
             "el vídeo mide 640×360 y dura 2 s. Es un vídeo sintético estático de cuatro fotogramas originales, "
             "del que el adaptador extrae 2 o 16 imágenes: no mide comprensión de movimiento ni vídeos complejos. "
             "Cada respuesta es una etiqueta de un token. La clasificación v2 utiliza 18 opciones y un prompt más largo; "
             "el control de texto corto y las modalidades utilizan tres opciones. No es una comparación de trabajos idénticos.", "",
             "El modelo permaneció cargado: no se mide su arranque. En el control se excluyó una inicialización del cliente Python "
             f"de {uncached['worker_initialization_request']['wall_seconds']:.2f} s para cargar el banco de recuperación. "
             "No se cambiaron pesos, parámetros del servicio, calibración ni versión activa v2. La caché de entrada se desactivó "
             "únicamente en el cliente de control y no en el servicio habitual.", "",
             "Fuentes metodológicas: [contadores PDH de Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw) "
             "y [opciones del servidor llama.cpp de esta revisión](https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/README.md).", "",
             f"Datos HTTP: {args.cached.resolve()}. Datos sin caché: {args.uncached.resolve()}. "
             "Ambos directorios contienen peticiones, respuestas, comprobaciones y todas las muestras de recursos."]
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    parts, in_table = [], False
    for line in text:
        if line.startswith("|---"):
            continue
        if line.startswith("|"):
            if not in_table:
                parts.append("<div class='scroll'><table>")
                in_table = True
            cells = line.strip("|").split("|")
            parts.append("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            parts.append("</table></div>")
            in_table = False
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line:
            parts.append(f"<p>{html.escape(line)}</p>")
    style = "body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;background:#f5f7fb;color:#152238}p{line-height:1.65}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}td{border:1px solid #d4dbe6;padding:12px;text-align:right}td:first-child{text-align:left}tr:first-child{font-weight:650;background:#e8eff9}.scroll{overflow:auto}h2{margin-top:36px}"
    (out / "report.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Consumo multimodal local</title><style>' + style + "</style>" + "".join(parts) + "</html>", encoding="utf-8")
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    benchmark = Path(__file__).with_name("benchmark_multimodal_resources.py")
    shutil.copy2(benchmark, out / benchmark.name)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
