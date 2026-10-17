"""Installer invariants without network, installed models or system changes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import stat
from zipfile import ZipFile, ZipInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


setup = module("windows_setup", "installer/windows/install.py")
builder = module("windows_builder", "scripts/build_windows_installer.py")


@pytest.mark.parametrize("name", ["../escape", "safe/../../escape", "/absolute", "C:/escape", "C:escape",
                                  "safe\\..\\escape", "file:stream", "//server/share/escape"])
def test_archive_cannot_escape_install_root(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("good.txt", "not written before validation completes")
        bundle.writestr(name, "bad")
    with pytest.raises(ValueError):
        setup.unpack(archive, tmp_path / "app")
    assert not (tmp_path / "app/good.txt").exists()
    assert not (tmp_path / "escape").exists()


def test_links_and_wheel_relocations_are_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    link = ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "../outside")
    with pytest.raises(ValueError):
        setup.unpack(archive, tmp_path / "app")
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("pkg.data/scripts/unexpected.exe", b"code")
    with pytest.raises(ValueError, match="relocation"):
        setup.unpack(archive, tmp_path / "site-packages", wheel=True)


def test_existing_link_inside_destination_cannot_redirect_writes(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "app"
    destination.mkdir()
    try:
        (destination / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlink creation for this account")
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("linked/escape.txt", "bad")
    with pytest.raises(ValueError):
        setup.unpack(archive, destination)
    assert not (outside / "escape.txt").exists()


def test_valid_wheel_keeps_package_and_license(tmp_path):
    archive = tmp_path / "package.whl"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("package/__init__.py", "VERSION = '1'")
        bundle.writestr("package.dist-info/licenses/LICENSE", "license")
    setup.unpack(archive, tmp_path / "packages", wheel=True)
    assert (tmp_path / "packages/package.dist-info/licenses/LICENSE").read_text() == "license"


def test_corrupted_download_never_replaces_a_verified_asset(tmp_path, monkeypatch):
    content = b"verified"
    spec = {"filename": "asset.zip", "url": "https://example.invalid/asset",
            "sha256": hashlib.sha256(content).hexdigest()}
    calls = []
    def response(*args, **kwargs):
        calls.append(True)
        return io.BytesIO(b"corrupted")
    monkeypatch.setattr(setup, "urlopen", response)
    monkeypatch.setattr(setup.time, "sleep", lambda seconds: None)
    with pytest.raises(ValueError, match="SHA-256"):
        setup.acquire(spec, tmp_path)
    assert len(calls) == 3 and not (tmp_path / "asset.zip").exists()
    monkeypatch.setattr(setup, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    asset = setup.acquire(spec, tmp_path)
    monkeypatch.setattr(setup, "urlopen", lambda *args, **kwargs: pytest.fail("verified asset redownloaded"))
    assert setup.acquire(spec, tmp_path) == asset


def test_wrong_external_cache_is_ignored_and_never_overwritten(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "asset.zip").write_bytes(b"wrong")
    good = b"good"
    spec = {"filename": "asset.zip", "url": "https://example.invalid/file",
            "sha256": hashlib.sha256(good).hexdigest()}
    monkeypatch.setattr(setup, "urlopen", lambda *args, **kwargs: io.BytesIO(good))
    path = setup.acquire(spec, tmp_path / "downloads", cache)
    assert path.read_bytes() == good and (cache / "asset.zip").read_bytes() == b"wrong"


def test_busy_ports_are_skipped_without_touching_the_existing_listener():
    # Windows can reserve large portions of its dynamic port range. Start with
    # a known usable lower pair instead of assuming adjacent ephemeral ports work.
    start, _ = setup.port_pair(20000, 40000)
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", start))
    occupied.listen()
    start = occupied.getsockname()[1]
    try:
        first, second = setup.port_pair(start, 40000)
        assert first > start and second == first + 1
        with socket.create_connection(("127.0.0.1", start), timeout=1):
            pass
    finally:
        occupied.close()


def test_dependency_lock_matches_e2b_download_and_uses_immutable_assets():
    lock = json.loads((ROOT / "installer/windows/dependencies.json").read_text())
    assert len(lock["wheels"]) >= 10
    for asset in [*lock["assets"].values(), *lock["wheels"]]:
        assert asset["url"].startswith("https://") and "/latest/" not in asset["url"]
        assert len(asset["sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in asset["sha256"])
    # Read source literals without mutating the shared E4B downloader module.
    import ast
    tree = ast.parse((ROOT / "scripts/download_gemma_e2b_qat.py").read_text())
    actual = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Attribute):
            if statement.targets[0].attr in ("REPO", "REVISION", "FILES"):
                actual[statement.targets[0].attr] = ast.literal_eval(statement.value)
    assert lock["model"]["repository"] == actual["REPO"]
    assert lock["model"]["revision"] == actual["REVISION"]
    assert lock["model"]["files"] == actual["FILES"]


def test_runtime_falls_back_to_cpu_when_vulkan_cannot_start(tmp_path, monkeypatch):
    import subprocess
    installed = []
    monkeypatch.setattr(setup, "install_asset", lambda spec, destination, *args: installed.append(destination))
    def invoke(arguments, **kwargs):
        if arguments[-1] == "--list-devices":
            raise subprocess.CalledProcessError(1, arguments)
        assert arguments[-1] == "--version"
        return subprocess.CompletedProcess(arguments, 0)
    monkeypatch.setattr(setup, "run", invoke)
    path, device = setup.runtime(tmp_path, {"vulkan": {}, "cpu": {}}, tmp_path / "downloads", None)
    assert device == "none" and "cpu" in path
    assert [directory.name for directory in installed] == ["llama-b11669-vulkan", "llama-b11669-cpu"]


def test_hidden_helper_output_is_streamed_and_failures_are_preserved(monkeypatch, capsys):
    import subprocess
    class Process:
        stdout = io.StringIO("25% descargado\nerror de descarga\n")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self):
            return 7
    monkeypatch.setattr(setup.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(subprocess.CalledProcessError) as error:
        setup.run(["python", "download.py"])
    assert error.value.returncode == 7
    assert "25% descargado" in capsys.readouterr().out


def test_upgrade_refuses_foreign_service_state_before_replacing_code(tmp_path, monkeypatch):
    from types import SimpleNamespace
    base, bundle = tmp_path / "installed", tmp_path / "bundle"
    base.mkdir()
    bundle.mkdir()
    (base / ".jev-qat-install.json").write_text(json.dumps({"product": setup.PRODUCT}))
    (base / "install.json").write_text(json.dumps({"source_sha256": "previous-version"}))
    state = base / "app/integration/jev-qat-e2b/service.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"status": "ready", "workspace": str(tmp_path / "other-installation")}))
    (bundle / "source.zip").write_bytes(b"new source not unpacked")
    (bundle / "bundle.json").write_text(json.dumps({"product": setup.PRODUCT,
        "source": {"sha256": setup.digest(bundle / "source.zip")}}))
    monkeypatch.setattr(setup, "run", lambda *args, **kwargs: pytest.fail("Foreign process stop attempted"))
    monkeypatch.setattr(setup, "unpack", lambda *args, **kwargs: pytest.fail("Code replaced before ownership check"))
    with pytest.raises(RuntimeError, match="otra instalacion"):
        setup.install(SimpleNamespace(install_dir=base, bundle=bundle))


def test_release_bundle_is_a_committed_snapshot_and_normalizes_batch_lines(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("pyproject.toml", '[project]\nversion="0.2.0"\n')
        archive.writestr("installer/windows/dependencies.json", json.dumps({"assets": {}, "wheels": []}))
        for filename in builder.BOOTSTRAP:
            archive.writestr("installer/windows/" + filename, "line1\nline2\n")
        for filename in ("LICENSE", "NOTICE.md", "installer/windows/LEEME.txt"):
            archive.writestr(filename, "documentation")
    source = stream.getvalue()
    monkeypatch.setattr(builder, "git", lambda *args: b"a" * 40 if args[0] in {"rev-parse", "rev-list"} else source)
    target = builder.build(tmp_path)
    with ZipFile(target) as bundle:
        manifest = json.loads(bundle.read("bundle.json"))
        assert manifest["source"]["commit"] == "a" * 40
        assert manifest["source"]["sha256"] == hashlib.sha256(source).hexdigest()
        assert bundle.read("source.zip") == source
        assert bundle.read("Instalar E2B.cmd") == b"line1\r\nline2\r\n"
    assert builder.ROOT.name not in target.with_suffix(".zip.sha256").read_text()
    assert setup.digest(target) in target.with_suffix(".zip.sha256").read_text()


def test_builder_refuses_private_artifacts_in_snapshot(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("integration/jev-qat/service.json", '{"secret":"not publishable"}')
    monkeypatch.setattr(builder, "git", lambda *args: b"a" * 40 if args[0] in {"rev-parse", "rev-list"} else stream.getvalue())
    with pytest.raises(ValueError, match="Unexpected"):
        builder.build(tmp_path)
