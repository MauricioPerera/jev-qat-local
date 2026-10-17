"""Launch/stop only the E2B instance belonging to this private installation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "stop"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    app = base / "app"
    record = json.loads((base / "install.json").read_text(encoding="utf8"))
    if record["status"] != "ready":
        raise RuntimeError("Ejecuta Instalar E2B para completar la descarga del modelo")
    os.environ["PATH"] = record["ffmpeg_bin"] + os.pathsep + os.environ.get("PATH", "")
    # The existing start() may reuse any healthy same-alias service on its port.
    # Check process ownership here so a second installation never takes it over.
    from jev_qat.e2b import RUNTIME_STATE, health
    if health():
        import psutil
        if not RUNTIME_STATE.is_file():
            raise RuntimeError("El puerto pertenece a otra instancia; revisa la configuracion")
        state = json.loads(RUNTIME_STATE.read_text(encoding="utf8"))
        if Path(state["workspace"]).resolve() != app or not psutil.pid_exists(state["pid"]):
            raise RuntimeError("El servicio pertenece a otra instalacion")
        process = psutil.Process(state["pid"])
        if abs(process.create_time() - state["created"]) >= .01 or "jev_qat.e2b" not in process.cmdline():
            raise RuntimeError("La identidad del proceso ha cambiado")
    subprocess.run([sys.executable, "-m", "jev_qat.e2b", args.command], cwd=app,
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if args.command == "start" and not args.no_browser:
        webbrowser.open(record["api_url"] + "/")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
