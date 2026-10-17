"""Build a small Windows installer ZIP from one committed Git snapshot."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ("Instalar E2B.cmd", "Instalar.ps1", "install.py", "launch.py")
ALLOWED_ROOTS = {".gitattributes", ".gitignore", ".github", "LICENSE", "NOTICE.md", "README.md",
                 "pyproject.toml", "jev_qat_config.json", "jev_qat_e2b_config.json", "jev_qat", "scripts",
                 "tests", "examples", "docs", "benchmarks", "installer"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def build(output, ref="HEAD"):
    commit = git("rev-parse", ref + "^{commit}").decode().strip()
    archive = git("archive", "--format=zip", commit)
    with ZipFile(io.BytesIO(archive)) as source:
        for item in source.infolist():
            if item.filename.split("/")[0] not in ALLOWED_ROOTS or item.file_size > 5 * 1024**2:
                raise ValueError("Unexpected or oversized file in source snapshot: " + item.filename)
        version = tomllib.loads(source.read("pyproject.toml").decode("utf8"))["project"]["version"]
        metadata = json.loads(source.read("installer/windows/dependencies.json"))
        metadata.update({"schema_version": 1, "product": "jev-qat-local-e2b", "version": version,
                         "source": {"commit": commit, "sha256": hashlib.sha256(archive).hexdigest()}})
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"JevQat-E2B-Windows-{version}.zip"
        if target.exists():
            raise FileExistsError("Use a new output directory; an installer already exists")
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as bundle:
            for filename in BOOTSTRAP:
                data = source.read("installer/windows/" + filename)
                if filename.endswith((".cmd", ".ps1")):
                    data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                bundle.writestr(filename, data)
            bundle.writestr("source.zip", archive)
            bundle.writestr("bundle.json", json.dumps(metadata, indent=2) + "\n")
            bundle.writestr("LEEME.txt", source.read("installer/windows/LEEME.txt"))
            bundle.writestr("LICENSE", source.read("LICENSE"))
            bundle.writestr("NOTICE.md", source.read("NOTICE.md"))
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n", encoding="ascii")
    print(json.dumps({"installer": str(target.resolve()), "sha256": checksum, "commit": commit}))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/windows")
    args = parser.parse_args()
    build(args.output, args.ref)


if __name__ == "__main__":
    main()
