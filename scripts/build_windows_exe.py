"""Compile an x64 single-file EXE around the committed Windows installer ZIP."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from zipfile import ZipFile


def compiler():
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Microsoft.NET"
    for architecture in ("Framework64", "Framework"):
        candidate = system / architecture / "v4.0.30319/csc.exe"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Building the EXE requires Windows with the .NET Framework C# compiler")


def compile_executable(payload, target, template, manifest, version):
    """Inputs are bytes from a fixed Git snapshot, never a moving remote branch."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("EXE version must have three numeric components")
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing installer EXE")
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(payload).hexdigest()
    code = template.decode("utf-8-sig").replace("@@VERSION@@", version).replace("@@PAYLOAD_SHA256@@", checksum)
    with tempfile.TemporaryDirectory(prefix="jev-qat-exe-build-") as directory:
        scratch = Path(directory)
        source = scratch / "Bootstrapper.cs"
        source.write_text(code, encoding="utf-8-sig")
        (scratch / "setup.manifest").write_bytes(manifest)
        (scratch / "payload.zip").write_bytes(payload)
        subprocess.run([
            str(compiler()), "/nologo", "/optimize+", "/target:exe", "/platform:x64",
            "/out:" + str(target), "/reference:System.IO.Compression.dll",
            "/reference:System.IO.Compression.FileSystem.dll",
            "/resource:" + str(scratch / "payload.zip") + ",JevQat.Payload.zip",
            "/win32manifest:" + str(scratch / "setup.manifest"), str(source),
        ], check=True, capture_output=True)
    # Confirm a real x64 PE was emitted, independently of its filename.
    with target.open("rb") as stream:
        header = stream.read(64)
        if header[:2] != b"MZ":
            raise ValueError("Compiler did not emit a Windows executable")
        stream.seek(struct.unpack_from("<I", header, 60)[0])
        if stream.read(6) != b"PE\0\0\x64\x86":
            raise ValueError("Installer must be an x64 PE executable")
    with target.open("rb") as stream:
        exe_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".exe.sha256").write_text(f"{exe_hash}  {target.name}\n", encoding="utf8")
    return target


def build_executable(bundle, output=None):
    bundle = Path(bundle)
    payload = bundle.read_bytes()
    with ZipFile(io.BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("bundle.json"))
        with ZipFile(io.BytesIO(archive.read("source.zip"))) as snapshot:
            template = snapshot.read("installer/windows/Bootstrapper.cs")
            manifest = snapshot.read("installer/windows/setup.manifest")
    target = Path(output or bundle.parent) / f"JevQat-E2B-Setup-{metadata['version']}.exe"
    compile_executable(payload, target, template, manifest, metadata["version"])
    print(json.dumps({"executable": str(target.resolve()), "commit": metadata["source"]["commit"],
                      "embedded_zip_sha256": hashlib.sha256(payload).hexdigest()}))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build_executable(args.bundle, args.output)


if __name__ == "__main__":
    main()
