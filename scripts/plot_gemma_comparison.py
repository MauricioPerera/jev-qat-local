"""Static figure from the measured E2B/E4B paired comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "inference" / "gemma-e2b-vs-e4b"


def main():
    report = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    rows = report["comparison"]
    labels = ["Texto v2", "Texto corto", "Foto 45,9 MP", "Foto 17,4 MP", "Voz 15 s", "Voz 30 s",
              "Vídeo 30 s · 2 f", "Vídeo 30 s · 16 f", "Vídeo 60 s · 2 f", "Vídeo 60 s · 16 f"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    metrics = [("median_seconds", "Latencia mediana", "segundos"),
               ("cpu_mean_percent", "CPU media, incluida la decodificación", "% de los 16 procesadores lógicos"),
               ("gpu_mean_percent", "GPU media durante la petición", "% del motor más ocupado"),
               ("gpu_shared_peak_gib", "Memoria gráfica compartida: pico", "GiB de RAM compartida")]
    y = np.arange(len(rows))
    for ax, (metric, title, unit) in zip(axes.flat, metrics):
        values = []
        for offset, model, color, name in ((-.18, "e4b", "#7b8fa8", "E4B"), (.18, "e2b", "#087f8c", "E2B")):
            data = [r[model][metric] for r in rows]
            values += data
            bars = ax.barh(y+offset, data, .34, color=color, label=name)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.18)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_xlabel(unit, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Gemma E2B frente a E4B · QAT Q4 oficial con componente multimodal\n"
                 "Mismas entradas reales · peticiones alternadas · cero tokens reutilizados", fontsize=15, weight="bold")
    fig.savefig(OUT / "comparison.png", dpi=170, facecolor="white")
    fig.savefig(OUT / "comparison.svg", facecolor="white")
    plt.close(fig)
    md = OUT / "report.md"
    if "## Gráfica comparativa" not in md.read_text(encoding="utf-8"):
        with md.open("a", encoding="utf-8") as stream:
            stream.write("\n## Gráfica comparativa\n\n![E2B frente a E4B]("+str((OUT/"comparison.png").resolve()).replace("\\","/")+")\n")
    html = OUT / "report.html"
    source = html.read_text(encoding="utf-8")
    if "<h2>Gráfica comparativa</h2>" not in source:
        source = source.replace("</html>", '<h2>Gráfica comparativa</h2><img src="/comparison/figure" alt="E2B frente a E4B" style="width:100%;height:auto"></html>')
        html.write_text(source, encoding="utf-8")
    print(str(OUT / "comparison.png"))


if __name__ == "__main__":
    main()
