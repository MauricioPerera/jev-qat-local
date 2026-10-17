"""Private Windows application installer; bootstrap uses only Python's stdlib."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

PRODUCT = "jev-qat-local-e2b"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def unpack(archive, destination, *, wheel=False):
    """Validate every entry before writing; never follow links or escape the root."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    with ZipFile(archive) as bundle:
        for item in bundle.infolist():
            name = item.filename.replace("\\", "/")
            mode = item.external_attr >> 16
            if (PureWindowsPath(name).drive or name.startswith("/") or ":" in name
                    or ".." in name.split("/") or stat.S_ISLNK(mode)):
                raise ValueError("Unsafe archive entry")
            if wheel and any(part.endswith(".data") for part in name.split("/")):
                raise ValueError("This installer supports wheels without .data relocation only")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive entry escapes installation directory")
            entries.append((item, target))
        for item, target in entries:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def acquire(spec, downloads, external_cache=None):
    filename = spec["filename"]
    if Path(filename).name != filename or PureWindowsPath(filename).drive:
        raise ValueError("Invalid asset filename")
    if not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
        raise ValueError("Invalid asset hash")
    target = Path(downloads) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == spec["sha256"]:
        return target
    if external_cache:
        cached = Path(external_cache) / filename
        if cached.is_file() and digest(cached) == spec["sha256"]:
            shutil.copyfile(cached, target)
            return target
    if not spec["url"].startswith("https://"):
        raise ValueError("Only HTTPS downloads are allowed")
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            print(f"Descargando {filename}...", flush=True)
            request = Request(spec["url"], headers={"User-Agent": "JevQatWindowsInstaller/0.2"})
            with urlopen(request, timeout=90) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if digest(partial) != spec["sha256"]:
                raise ValueError(f"SHA-256 incorrecto: {filename}")
            partial.replace(target)
            return target
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def install_asset(spec, destination, downloads, external_cache=None):
    receipt = Path(destination) / ".archive-sha256"
    if receipt.is_file() and receipt.read_text().strip() == spec["sha256"]:
        return
    archive = acquire(spec, downloads, external_cache)
    unpack(archive, destination)
    receipt.write_text(spec["sha256"], encoding="ascii")


def port_pair(first=18166, last=18366):
    for backend in range(first, last, 2):
        probes = []
        try:
            for port in (backend, backend + 1):
                probe = socket.socket()
                probes.append(probe)
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", port))
            return backend, backend + 1
        except OSError:
            pass
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError("No hay un par de puertos locales disponible")


def run(arguments, **kwargs):
    return subprocess.run([str(arg) for arg in arguments], check=True,
                          creationflags=NO_WINDOW, **kwargs)


def runtime(app, assets, downloads, external_cache):
    vulkan = app / "runtimes/llama-b11669-vulkan"
    install_asset(assets["vulkan"], vulkan, downloads, external_cache)
    try:
        result = run([vulkan / "llama-server.exe", "--list-devices"],
                     capture_output=True, text=True, encoding="utf8", errors="replace", timeout=60)
        device = re.search(r"^\s*(Vulkan\d+):", result.stdout + result.stderr, re.MULTILINE)
        if device:
            return "runtimes/llama-b11669-vulkan/llama-server.exe", device[1]
    except (OSError, subprocess.SubprocessError):
        pass
    print("Vulkan no disponible; configurando motor CPU (mas lento).", flush=True)
    cpu = app / "runtimes/llama-b11669-cpu"
    install_asset(assets["cpu"], cpu, downloads, external_cache)
    run([cpu / "llama-server.exe", "--version"], capture_output=True, timeout=60)
    return "runtimes/llama-b11669-cpu/llama-server.exe", "none"


def shortcuts(base):
    # WScript.Shell shortcut creation uses structured arguments through a script,
    # with paths supplied via environment rather than interpolated command text.
    script = base / "shortcuts.ps1"
    script.write_text("""$ErrorActionPreference = 'Stop'
$folder = Join-Path ([Environment]::GetFolderPath('Programs')) 'Jev QAT E2B'
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$shell = New-Object -ComObject WScript.Shell
foreach ($name in @('Abrir E2B', 'Detener E2B')) {
    $link = $shell.CreateShortcut((Join-Path $folder ($name + '.lnk')))
    $link.TargetPath = Join-Path $env:JEV_INSTALL_BASE ($name + '.cmd')
    $link.WorkingDirectory = $env:JEV_INSTALL_BASE
    $link.Save()
}
""", encoding="ascii")
    env = {**os.environ, "JEV_INSTALL_BASE": str(base)}
    run([os.environ.get("SystemRoot", r"C:\Windows") + r"\System32\WindowsPowerShell\v1.0\powershell.exe",
         "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script], env=env)


def install(args):
    base, bundle_dir = args.install_dir.resolve(), args.bundle.resolve()
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf8"))
    owner = json.loads((base / ".jev-qat-install.json").read_text(encoding="utf8"))
    if bundle["product"] != PRODUCT or owner["product"] != PRODUCT:
        raise ValueError("Installation directory is not owned by this application")
    app, downloads = base / "app", base / "downloads"
    app.mkdir(exist_ok=True)
    downloads.mkdir(exist_ok=True)
    source = bundle_dir / "source.zip"
    if digest(source) != bundle["source"]["sha256"]:
        raise ValueError("SHA-256 del codigo incorrecto")
    previous = json.loads((base / "install.json").read_text(encoding="utf8")) if (base / "install.json").exists() else {}
    config_path = app / "jev_qat_e2b_config.json"
    config = json.loads(config_path.read_text(encoding="utf8")) if config_path.exists() else None
    if previous.get("source_sha256") != bundle["source"]["sha256"]:
        if (app / "integration/jev-qat-e2b/service.json").exists():
            # Never hot-replace a possibly active application.
            state = json.loads((app / "integration/jev-qat-e2b/service.json").read_text(encoding="utf8"))
            if state.get("status") in {"ready", "starting"}:
                raise RuntimeError("Deten E2B antes de instalar otra version")
        unpack(source, app)
    packages = base / "python/Lib/site-packages"
    wheel_receipts = packages / ".installed-wheels"
    for wheel in bundle["wheels"]:
        receipt = wheel_receipts / (wheel["filename"] + ".sha256")
        if receipt.is_file() and receipt.read_text().strip() == wheel["sha256"]:
            continue
        archive = acquire(wheel, downloads, args.download_cache)
        unpack(archive, packages, wheel=True)
        wheel_receipts.mkdir(parents=True, exist_ok=True)
        receipt.write_text(wheel["sha256"], encoding="ascii")
    run([sys.executable, "-c", "import httpx,fastapi,uvicorn,pydantic,psutil; from jev_qat.e2b import E2BQatAdapter; print('Dependencias verificadas.')"])
    selected_runtime, device = runtime(app, bundle["assets"], downloads, args.download_cache)
    codecs = base / "tools/ffmpeg"
    install_asset(bundle["assets"]["ffmpeg"], codecs, downloads, args.download_cache)
    ffmpeg = list(codecs.rglob("ffmpeg.exe"))
    if len(ffmpeg) != 1 or not (ffmpeg[0].parent / "ffprobe.exe").is_file():
        raise RuntimeError("No se encontraron ffmpeg y ffprobe")
    for program in (ffmpeg[0], ffmpeg[0].parent / "ffprobe.exe"):
        run([program, "-version"], capture_output=True, timeout=30)
    if config is None:
        config = json.loads(config_path.read_text(encoding="utf8"))
        config["backend_port"], config["api_port"] = port_pair()
        config["threads"] = min(8, os.cpu_count() or 4)
    config["runtime"], config["device"] = selected_runtime, device
    save_json(config_path, config)
    record = {"product": PRODUCT, "version": bundle["version"], "commit": bundle["source"]["commit"],
              "source_sha256": bundle["source"]["sha256"], "ffmpeg_bin": str(ffmpeg[0].parent),
              "api_url": f"http://127.0.0.1:{config['api_port']}", "device": device,
              "status": "dependencies_ready"}
    save_json(base / "install.json", record)
    shutil.copyfile(bundle_dir / "launch.py", base / "launch.py")
    for name, command in (("Abrir E2B", "start"), ("Detener E2B", "stop")):
        (base / (name + ".cmd")).write_text(
            '@echo off\r\nsetlocal DisableDelayedExpansion\r\n"%~dp0python\\python.exe" "%~dp0launch.py" '
            + command + '\r\nif errorlevel 1 pause\r\n', encoding="ascii")
    if not args.no_shortcuts:
        shortcuts(base)
    if args.dependencies_only:
        print("Entorno preparado. Ejecuta de nuevo sin -DependenciesOnly para descargar el modelo.", flush=True)
        return
    missing_bytes = sum(f["size"] for name, f in bundle["model"]["files"].items()
                        if not (app / bundle["model"]["directory"] / name).exists())
    if shutil.disk_usage(base).free < missing_bytes + 256 * 1024**2:
        raise RuntimeError("Espacio insuficiente para el modelo y su componente multimodal")
    if args.model_cache:
        for name, spec in bundle["model"]["files"].items():
            source_model = args.model_cache / name
            target = app / bundle["model"]["directory"] / name
            if not target.exists() and source_model.is_file() and digest(source_model) == spec["sha256"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".copy-partial")
                shutil.copyfile(source_model, temporary)
                if digest(temporary) != spec["sha256"]:
                    raise ValueError("Model cache changed during copying")
                temporary.replace(target)
    print("Verificando/descargando E2B QAT Q4 y su componente multimodal (4,34 GB)...", flush=True)
    run([sys.executable, app / "scripts/download_gemma_e2b_qat.py"], cwd=app)
    record["status"] = "ready"
    save_json(base / "install.json", record)
    if not args.no_launch:
        command = [sys.executable, base / "launch.py", "start"]
        if args.no_browser:
            command.append("--no-browser")
        run(command, cwd=app)
    print(f"Instalado en: {base}\nInterfaz: {record['api_url']}/", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--download-cache", type=Path)
    parser.add_argument("--model-cache", type=Path)
    for flag in ("dependencies-only", "no-launch", "no-browser", "no-shortcuts"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    try:
        install(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
