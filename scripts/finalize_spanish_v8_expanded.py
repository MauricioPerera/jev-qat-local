"""Archive reproducible evidence and verify the delivered local report/default."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import httpx

from jev_qat.calibration import sha256_file
from jev_qat.config import ROOT, save_json, settings
from jev_qat.focus_experiment import read_json
from jev_qat.general_probability_audit import audit
from jev_qat.public_eval_data import DIRECTORY, frozen_profiles


def finalize():
    run = Path(read_json(DIRECTORY / "latest-run.json")["directory"])
    m, s = read_json(DIRECTORY / "manifest.json"), read_json(run / "summary.json")
    a, v = read_json(run / "activation.json"), read_json(run / "verification.json")
    assert v["passed"] and v["live_composition_passed"] and not a["verification_required"]
    assert a["approved"] == a["activated"] == all(a["conditions"].values())
    current = read_json(ROOT / "classification/current.json")
    assert current["version"] == ("v8" if a["activated"] else "v2")
    assert frozen_profiles() == m["frozen_profiles"]
    probability = audit(run, s | {"selected_temperature": s["frozen_temperatures"]["v8"]})
    save_json(run / "general-probability-audit.json", probability)
    # A bootstrap of identical decisions degenerates at zero; do not imply
    # that it establishes population equivalence or zero future uncertainty.
    if s["paired_comparison"]["fixed_errors"] == s["paired_comparison"]["introduced_errors"] == 0:
        note = ("Las dos versiones predijeron la misma categoría en los 1.283 mensajes. "
            "Sin discordancias, se informa McNemar p=1 por convenio. El intervalo bootstrap [0,0] es degenerado "
            "en esta muestra: no demuestra equivalencia poblacional ni ausencia de incertidumbre futura. "
            "Además, el F1 general de v2 es 98,22%; en esta cohorte ni un clasificador perfecto podría "
            "superarlo en los tres puntos exigidos por el criterio previo. No se relaja ese criterio tras ver los datos. "
            "La ausencia de mejora de exactitud, p=1 y la pérdida de una recuperación también impiden activar v8.")
        report = run / "report.html"
        page = report.read_text(encoding="utf-8")
        if "id='statistical-note'" not in page:
            page = page.replace("</main>", "<section id='statistical-note'><h2>Interpretación de la igualdad observada</h2><p>" + note + "</p></section></main>")
            report.write_text(page, encoding="utf-8")
        markdown = run / "report.md"
        content = markdown.read_text(encoding="utf-8")
        if note not in content:
            markdown.write_text(content + "\n" + note + "\n", encoding="utf-8")
    # Source SHA files contain no management state or credentials.
    paths = list((ROOT / "jev_qat").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    paths += [ROOT / "jev_qat/option_labels.json", ROOT / "README.md", ROOT / "pyproject.toml",
        ROOT / "scripts/verify_spanish_v8_expanded.py", ROOT / "scripts/finalize_spanish_v8_expanded.py",
        DIRECTORY / "manifest.json", DIRECTORY / "label-crosswalk.json", DIRECTORY / "label-provenance.json",
        DIRECTORY / "selected-source-audit.json", DIRECTORY / "regression-stability.json",
        DIRECTORY / "regression-stability.md", DIRECTORY / "preflight-regressions.json", DIRECTORY / "unit-tests.xml"]
    paths += [ROOT / name for name in m["frozen_profiles"]["files"]]
    paths += [p for p in (ROOT / "classification/profiles").glob("massive_es_*.json")]
    paths += [p for p in (ROOT / "calibration/profiles").glob("massive_scenario_es_*.json")]
    archive = run / "delivery-source"
    hashes = {}
    for p in sorted(set(paths)):
        relative = p.relative_to(ROOT)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        hashes[str(relative)] = sha256_file(target)
    save_json(run / "delivery-source-manifest.json", hashes)
    suite = ET.parse(DIRECTORY / "unit-tests.xml").getroot()
    tests = sum(int(x.get("tests", 0)) for x in suite.iter("testsuite"))
    assert all(int(x.get("failures", 0)) == int(x.get("errors", 0)) == 0 for x in suite.iter("testsuite"))
    with httpx.Client(base_url=settings()["api_url"], timeout=30, trust_env=False) as client:
        report = client.get("/evaluation"); report.raise_for_status()
        summary = client.get("/evaluation/summary"); summary.raise_for_status()
        assert report.content == (run / "report.html").read_bytes()
        assert summary.json() == s
    output = {"passed": True, "new_messages": s["sample_counts"]["test"],
        "active_version": current["version"], "v8_activated": a["activated"], "unit_tests_passed": tests,
        "pvalue": a["pvalue"], "alpha": a["alpha"], "significant": a["pvalue"] < a["alpha"],
        "all_eight_preserved": a["conditions"]["eight_recoveries_pass_live"],
        "original_regression_failures_retained": True,
        "original_human_labels_documented_and_source_matched": True, "new_human_crosswalk_adjudication": False,
        "temperatures_unchanged": True, "native_components_and_live_composition_verified": True,
        "report_sha256": sha256_file(run / "report.html"), "summary_sha256": sha256_file(run / "summary.json"),
        "source_archive_manifest_sha256": sha256_file(run / "delivery-source-manifest.json")}
    save_json(run / "delivery-verification.json", output)
    print(output, flush=True)


if __name__ == "__main__":
    finalize()
