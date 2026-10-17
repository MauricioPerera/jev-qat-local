"""Verify explicit v7, previous v6 availability and the selected default helper."""
from pathlib import Path

from jev_qat import classify_spanish_message
from jev_qat.config import ROOT, save_json
from jev_qat.focus_experiment import read_json
from jev_qat.general_precision_data import DIRECTORY


def main():
    run = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    calibrated = read_json(run/"selection.json")["validation_approved"]
    results = []
    for message,expected in (("Hola, ¿cómo estás?","general"),("Añade arroz a mi lista de compras","lists"),("Léeme los titulares del periódico de hoy","news")):
        result = classify_spanish_message(message,version="v7",calibrated=calibrated)
        assert result["category"] == expected and result["metadata"]["task_profile"] == "massive_es_v7"
        assert result["metadata"]["calibrated"] == calibrated
        results.append({"message":message,"result":result})
    default_version = read_json(ROOT/"classification/current.json")["version"]
    default = classify_spanish_message("Hola, ¿cómo estás?")
    assert default["category"] == "general" and default["metadata"]["calibrated"]
    assert default["metadata"]["task_profile"] == f"massive_es_{default_version}"
    previous = classify_spanish_message("Hola, ¿cómo estás?",version="v6")
    assert previous["category"] == "general" and previous["metadata"]["calibrated"]
    if not read_json(run/"activation.json")["approved"]:
        assert default_version == "v2"
    save_json(run/"helper-smoke.json",{"passed":True,"v7":results,"default_version":default_version,"default":default,"previous_v6":previous})
    print(f"Public helper passed; default remains {default_version}.")


if __name__ == "__main__":
    main()
