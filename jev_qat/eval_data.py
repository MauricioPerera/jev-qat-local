"""Pinned public Spanish MASSIVE data and a frozen, leakage-checked protocol."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import unicodedata

import httpx

from .calibration import PROMPT_VERSION, sha256_file
from .config import ROOT, save_json

DIRECTORY = ROOT / "evaluation" / "massive-es-v1"
REVISION = "58871793b91addb7c5f7afff26ccf08737fb6697"
DATASET = "mteb/amazon_massive_scenario"
SEED = 20261016
QUESTION = "¿A qué categoría pertenece la solicitud del usuario? Elige la categoría más específica."
CATEGORIES = {
    "alarm": "Alarmas: crear, cambiar, consultar o cancelar una alarma o despertador.",
    "audio": "Volumen y sonido del dispositivo: subir, bajar o silenciar el audio.",
    "calendar": "Calendario: reuniones, citas, eventos y recordatorios con fecha u hora.",
    "cooking": "Cocina: recetas, ingredientes, preparación y tiempos de cocción.",
    "datetime": "Fecha y hora: hora actual, fechas, días y zonas horarias.",
    "email": "Correo electrónico: enviar, leer, buscar o responder correos y mensajes de email.",
    "general": "Conversación general con el asistente: saludos, bromas, ayuda, respuestas y otras solicitudes generales.",
    "iot": "Hogar inteligente: controlar luces, enchufes, calefacción y otros aparatos conectados.",
    "lists": "Listas: crear, consultar o modificar listas de compras, tareas y pendientes.",
    "music": "Información y preferencias musicales: identificar canciones o artistas, gustos y listas favoritas.",
    "news": "Noticias: consultar titulares, actualidad y noticias por tema.",
    "play": "Reproducción de contenido: poner, buscar para reproducir o controlar música, radio, podcasts, audiolibros y juegos.",
    "qa": "Preguntas de conocimiento: datos, definiciones, cálculos, divisas, cotizaciones y otras consultas de información.",
    "recommendation": "Recomendaciones: sugerir restaurantes, lugares, eventos, películas y actividades.",
    "social": "Redes sociales: publicar, leer o interactuar en Twitter, Facebook y otras redes.",
    "takeaway": "Comida a domicilio o para llevar: pedir comida, gestionar pedidos y consultar su estado.",
    "transport": "Transporte y viajes: rutas, tráfico, direcciones, taxis, trenes y reservas de viaje.",
    "weather": "Tiempo meteorológico: clima, temperatura, lluvia y pronóstico del tiempo.",
}
SOURCE_HASHES = {
    "train": "1a641bde768424425863a7df4f3fe24701a9341c0751a198c688186aeee597c5",
    "validation": "7df17188ea1df2adead7ffe9e864a9caaec89dae78a7a292cfc90aa9b939c6c5",
    "test": "1e835f4bce1505e07e5c714f597255fc6ed3c930432fcd7c76d18af8b62448d2",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def text_group(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def stratified_sample(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    if count < len(groups) or count > len(rows):
        raise ValueError("Sample size must cover every class and fit the available data")
    quotas = {label: max(1, int(count*len(values)/len(rows))) for label, values in groups.items()}
    while sum(quotas.values()) < count:
        label = max(groups, key=lambda key: (count*len(groups[key])/len(rows)-quotas[key], key))
        quotas[label] += 1
    while sum(quotas.values()) > count:
        label = max((key for key in groups if quotas[key] > 1),
                    key=lambda key: (quotas[key]-count*len(groups[key])/len(rows), key))
        quotas[label] -= 1
    selected = []
    for label in sorted(groups):
        values = sorted(groups[label], key=lambda row: str(row["id"]))
        random.Random(f"{seed}:{label}").shuffle(values)
        selected.extend(values[:quotas[label]])
    random.Random(seed).shuffle(selected)
    return selected


def disjoint_sources(sources):
    """Drop conflicting texts; reserve duplicate groups for the latest official split."""
    labels = defaultdict(set)
    for rows in sources.values():
        for row in rows:
            labels[text_group(row["text"])].add(row["label"])
    ambiguous = {key for key, values in labels.items() if len(values) > 1}
    seen_text, seen_ids = set(), set()
    result, dropped = {}, {}
    for split in ("test", "validation", "train"):
        kept = []
        for row in sources[split]:
            group = text_group(row["text"])
            if group in ambiguous or group in seen_text or str(row["id"]) in seen_ids:
                continue
            if row["label"] not in CATEGORIES or not row["text"].strip() or row["lang"] != "es":
                raise ValueError("Unexpected source schema, language or category")
            kept.append(row)
            seen_text.add(group)
            seen_ids.add(str(row["id"]))
        result[split] = kept
        dropped[split] = len(sources[split])-len(kept)
    return result, {"ambiguous_text_groups": len(ambiguous), "removed_by_split": dropped}


def prepare(directory=DIRECTORY):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split, item in manifest["splits"].items():
            if sha256_file(directory / item["file"]) != item["sha256"]:
                raise ValueError(f"Frozen {split} data changed; use a new evaluation directory")
        if manifest["protocol"]["options"] != CATEGORIES or manifest["protocol"]["question"] != QUESTION:
            raise ValueError("Frozen protocol differs from code; use a new version")
        return manifest
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    sources, downloads = {}, {}
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for split, expected in SOURCE_HASHES.items():
            path = source_dir / f"{split}-es.json.gz"
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{split}/es.json.gz"
            if not path.exists():
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
            if sha256_file(path) != expected:
                raise ValueError(f"Source checksum mismatch for {split}")
            sources[split] = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines() if line]
            downloads[split] = {"url": url, "sha256": expected, "bytes": path.stat().st_size, "rows": len(sources[split])}
        card = source_dir / "README.md"
        if not card.exists():
            response = client.get(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/README.md")
            response.raise_for_status()
            card.write_bytes(response.content)
    cleaned, removals = disjoint_sources(sources)
    splits = {}
    for index, (split, source, count) in enumerate((("calibration", "train", 300), ("validation", "validation", 150), ("test", "test", 450))):
        selected = stratified_sample(cleaned[source], count, SEED+index)
        records = []
        for row in selected:
            labels = list(CATEGORIES)
            random.Random(f"{SEED}:options:{source}:{row['id']}").shuffle(labels)
            records.append({
                "id": f"massive:es:{source}:{row['id']}", "source_id": str(row["id"]),
                "group_id": text_group(row["text"]), "source_split": source,
                "state": row["text"], "question": QUESTION,
                "options": [CATEGORIES[label] for label in labels], "option_labels": labels,
                "target": row["label"], "modality": "text",
            })
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records), encoding="utf-8")
        splits[split] = {"file": path.name, "sha256": sha256_file(path), "count": count,
                         "class_counts": dict(sorted(Counter(row["target"] for row in records).items()))}
    manifest = {
        "schema_version": 1, "dataset": DATASET, "revision": REVISION, "language": "es",
        "source": downloads, "splits": splits, "deduplication": removals,
        "protocol": {"seed": SEED, "question": QUESTION, "options": CATEGORIES,
                     "prompt_version": PROMPT_VERSION, "option_order": "deterministic shuffle per source record",
                     "sampling": "proportional stratified; official train/validation/test; no shared normalized text or source ID",
                     "fit": "scalar temperature minimizing calibration NLL, T bounded to [0.05,100]",
                     "activation_gate": "validation NLL improves by more than 0.0001; test never selects temperature or activation",
                     "order_audit_count": 36, "metrics": ["accuracy", "macro_f1", "NLL", "multiclass Brier sum", "top-label ECE (10 equal-width bins)"]},
        "attribution": {
            "original": "MASSIVE: FitzGerald et al. (2022), Amazon Science / Alexa, https://github.com/alexa/massive",
            "paper": "https://arxiv.org/abs/2204.08582", "original_data_license": "CC BY 4.0",
            "mirror": f"https://huggingface.co/datasets/{DATASET}", "mirror_card_license": "Apache-2.0",
            "data_description": "Human-localized assistant requests; public benchmark, not private production traffic.",
        },
    }
    manifest["manifest_fingerprint"] = digest(manifest)
    save_json(manifest_path, manifest)
    return manifest


def load_records(directory, split):
    return [json.loads(line) for line in (Path(directory) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
