import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "integration" / "jev-qat"
RUNTIME_STATE = STATE / "service.json"


def settings():
    config = json.loads((ROOT / "jev_qat_config.json").read_text(encoding="utf-8"))
    for field in ("runtime", "model", "projector"):
        config[field] = str((ROOT / config[field]).resolve())
    config["media_roots"] = [str((ROOT / value).resolve()) for value in config["media_roots"]]
    config["backend_url"] = f"http://127.0.0.1:{config['backend_port']}"
    config["api_url"] = f"http://127.0.0.1:{config['api_port']}"
    return config


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
