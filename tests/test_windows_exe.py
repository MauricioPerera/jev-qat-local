"""Execute the compiled launcher: integrity, Unicode args and process lifetime."""
import importlib.util
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exe_builder", ROOT / "scripts/build_windows_exe.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Native Windows EXE tests")


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    folder = tmp_path_factory.mktemp("exe con espacios ñ")
    script = r'''param([string]$InstallDir, [string]$ModelCache, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
while (-not (Test-Path -LiteralPath $env:JEV_SETUP_GATE)) { Start-Sleep -Milliseconds 20 }
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$info = New-Object Diagnostics.ProcessStartInfo
$info.FileName = Join-Path $PSHOME 'powershell.exe'
$info.Arguments = '-NoProfile -Command "Start-Sleep -Seconds 60"'
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$child = [Diagnostics.Process]::Start($info)
$record = @{pid=$child.Id; created=$child.StartTime.ToUniversalTime().ToString('o'); path=$InstallDir; mode=$ModelCache; no_browser=[bool]$NoBrowser}
[IO.File]::WriteAllText((Join-Path $InstallDir 'child.json'), ($record | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
Write-Output 'MOCK_SETUP_READY'
if ($ModelCache -eq 'fail') { exit 7 }
if ($ModelCache -eq 'wait') { Start-Sleep -Seconds 60 }
exit 0
'''
    payload = io.BytesIO()
    names = ("Instalar E2B.cmd", "Instalar.ps1", "install.py", "launch.py", "source.zip", "bundle.json", "LEEME.txt", "LICENSE", "NOTICE.md")
    with ZipFile(payload, "w") as bundle:
        for name in names:
            bundle.writestr(name, script if name == "Instalar.ps1" else "fixture")
    target = folder / "Instalador E2B ñ.exe"
    builder.compile_executable(payload.getvalue(), target,
        (ROOT / "installer/windows/Bootstrapper.cs").read_bytes(),
        (ROOT / "installer/windows/setup.manifest").read_bytes(), "0.3.0")
    return target, payload.getvalue()


def alive(pid):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def wait_dead(pid):
    for _ in range(100):
        if not alive(pid):
            return
        time.sleep(.05)
    pytest.fail("Installer-owned child was left alive")


def test_exe_verifies_embedded_payload_without_installation(executable):
    exe, _ = executable
    result = subprocess.run([str(exe), "--verify", "--non-interactive"], capture_output=True, text=True, encoding="utf8", timeout=20)
    assert result.returncode == 0 and "PAYLOAD_OK version=0.3.0" in result.stdout
    assert exe.with_suffix(".exe.sha256").is_file()


def test_corrupt_embedded_payload_is_rejected_before_running_scripts(executable, tmp_path):
    exe, payload = executable
    data = bytearray(exe.read_bytes())
    start = data.find(payload)
    assert start >= 0
    data[start + 100] ^= 1
    corrupt = tmp_path / "corrupt.exe"
    corrupt.write_bytes(data)
    result = subprocess.run([str(corrupt), "--verify", "--non-interactive"], capture_output=True, text=True, encoding="utf8", timeout=20)
    assert result.returncode != 0 and "SHA-256" in result.stdout


def test_exe_preserves_successful_service_and_unicode_arguments(executable, tmp_path):
    exe, _ = executable
    destination = tmp_path / "Instalación ñ con espacios & [datos] ! %"
    # A deliberately preserved fixture child may retain inherited pipe handles.
    # Wait for the installer process itself, independently of descendant EOF.
    log = tmp_path / "setup.log"
    with log.open("wb") as output:
        result = subprocess.run([str(exe), "-InstallDir", str(destination), "-ModelCache", "success",
                                 "-NoBrowser", "--non-interactive"], stdout=output, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, log.read_text(encoding="utf8")
    child = json.loads((destination / "child.json").read_text(encoding="utf8"))
    process = psutil.Process(child["pid"])
    try:
        assert child["path"] == str(destination) and child["no_browser"] is True
        assert child["mode"] == "success" and process.is_running()
    finally:
        # Only terminate the just-created fixture sleeper, never unrelated processes.
        created = datetime.fromisoformat(child["created"].replace("Z", "+00:00")).timestamp()
        if (process.is_running() and abs(process.create_time() - created) < .05
                and process.name().lower() == "powershell.exe" and "Start-Sleep" in " ".join(process.cmdline())):
            process.terminate()
            process.wait(timeout=10)


def test_failed_install_kills_its_own_descendants_and_returns_failure(executable, tmp_path):
    exe, _ = executable
    destination = tmp_path / "failed"
    result = subprocess.run([str(exe), "-InstallDir", str(destination), "-ModelCache", "fail", "--non-interactive"],
                            capture_output=True, text=True, encoding="utf8", timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
    child = json.loads((destination / "child.json").read_text())
    wait_dead(child["pid"])


def test_closing_incomplete_exe_kills_only_its_process_tree(executable, tmp_path):
    exe, _ = executable
    destination = tmp_path / "cancelled"
    process = subprocess.Popen([str(exe), "-InstallDir", str(destination), "-ModelCache", "wait", "--non-interactive"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if (destination / "child.json").exists():
                break
            time.sleep(.05)
        else:
            pytest.fail("Fixture installer did not start")
        child = json.loads((destination / "child.json").read_text())
        assert alive(child["pid"])
        process.kill()
        process.wait(timeout=10)
        wait_dead(child["pid"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if process.stdout:
            process.stdout.close()
