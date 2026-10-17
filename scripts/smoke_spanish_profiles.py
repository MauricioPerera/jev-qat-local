"""Exercise the public helper, calibrated default and experimental selective task."""
from pathlib import Path

from jev_qat import classify_spanish_message
from jev_qat.config import ROOT, save_json
from jev_qat.focus_experiment import read_json
from jev_qat.selective_experiment import DIRECTORY


def main():
    results = []
    for message,expected in (("Hola, ¿cómo estás?","general"),
                             ("Añade arroz a mi lista de compras","lists"),
                             ("Léeme los titulares del periódico de hoy","news")):
        result = classify_spanish_message(message,version="v4",calibrated=False)
        assert result["category"] == expected
        assert result["metadata"]["task_profile"] == "massive_es_v4" and not result["metadata"]["calibrated"]
        results.append({"message":message,"response":result})
    default = classify_spanish_message("Añade arroz a mi lista de compras")
    assert default["category"] == "lists"
    assert default["metadata"]["task_profile"] == "massive_es_v2" and default["metadata"]["calibrated"]
    run = Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    save_json(run/"helper-smoke.json",{"passed":True,"v4":results,"default_v2":default})
    print("Helper: three v4 focus messages correct; calibrated v2 default preserved.")


if __name__=="__main__":
    main()
