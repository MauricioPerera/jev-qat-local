"""Verify the installed distribution can carry its tokenizer labels and UI."""
from pathlib import Path
from zipfile import ZipFile


def main():
    wheels = list(Path("dist").glob("jev_qat_local-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one built jev_qat_local wheel in dist/")
    expected = {
        "jev_qat/option_labels.json", "jev_qat/web/e2b.html",
        "jev_qat/web/e2b.css", "jev_qat/web/e2b.js",
    }
    with ZipFile(wheels[0]) as archive:
        missing = expected - set(archive.namelist())
        if missing:
            raise SystemExit(f"Missing package assets: {sorted(missing)}")
    print("Wheel includes all four tokenizer/UI assets.")


if __name__ == "__main__":
    main()
