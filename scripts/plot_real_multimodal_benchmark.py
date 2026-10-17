"""Standalone resource comparison figure from measured results only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "real-media-benchmark-20261017"


def main():
    report = json.loads((OUT / "run-01" / "results.json").read_text(encoding="utf-8"))
    controls = {r["case"]: r for r in report["comparison"] if r["kind"] == "legacy_control"}
    real = [r for r in report["comparison"] if r["kind"] == "real_media"]
    labels = ["Foto 45,9 MP", "Foto 17,4 MP", "Voz 15 s", "Voz 30 s", "Vídeo 30 s · 2 f", "Vídeo 30 s · 16 f", "Vídeo 60 s · 2 f", "Vídeo 60 s · 16 f"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    settings = [("median_seconds", "Latencia mediana", "segundos"),
                ("cpu_total_mean_percent", "CPU media: modelo, adaptador y decodificación", "% de los 16 procesadores lógicos"),
                ("gpu_mean_percent", "GPU media durante la petición", "% del motor más ocupado"),
                ("rss_peak_gib", "RAM residente: pico muestreado de los procesos", "GiB de conjuntos de trabajo de los procesos")]
    y = np.arange(len(real))
    for axis, (metric, title, unit) in zip(axes.flat, settings):
        baseline = [controls[r["reference_case"]][metric] for r in real]
        values = [r[metric] for r in real]
        old_bars = axis.barh(y - .18, baseline, .34, label="Archivo breve, repetido en esta sesión", color="#a8b6c7")
        new_bars = axis.barh(y + .18, values, .34, label="Contenido real de mayor tamaño/duración", color="#087f8c")
        axis.set_yticks(y, labels, fontsize=9)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=11, weight="bold")
        axis.set_xlabel(unit, fontsize=9)
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0, max(baseline + values) * 1.21)
        axis.bar_label(old_bars, fmt="%.2f", padding=3, fontsize=7, color="#455a70")
        axis.bar_label(new_bars, fmt="%.2f", padding=3, fontsize=7)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend, loc="outside lower center", ncols=2, frameon=False, fontsize=10)
    figure.suptitle("Gemma E4B QAT Q4 · medios reales · cero tokens de entrada reutilizados\n"
                    "Cuatro pasadas por caso; tiempos y medias de las tres posteriores a la primera", fontsize=15, weight="bold")
    figure.savefig(OUT / "comparison.png", dpi=170, facecolor="white")
    figure.savefig(OUT / "comparison.svg", facecolor="white")
    plt.close(figure)
    link = "\n## Gráfica comparativa\n\n![Comparación con archivos breves en la misma sesión](" + str((OUT / "comparison.png").resolve()).replace("\\", "/") + ")\n"
    markdown = OUT / "report.md"
    if "## Gráfica comparativa" not in markdown.read_text(encoding="utf-8"):
        with markdown.open("a", encoding="utf-8") as stream:
            stream.write(link)
    document = OUT / "report.html"
    source = document.read_text(encoding="utf-8")
    if "<h2>Gráfica comparativa</h2>" not in source:
        source = source.replace("</html>", '<h2>Gráfica comparativa</h2><img src="comparison.png" alt="Comparación de consumo y latencia" style="width:100%;height:auto"></html>')
    document.write_text(source, encoding="utf-8")
    print(str(OUT / "comparison.png"))


if __name__ == "__main__":
    main()
