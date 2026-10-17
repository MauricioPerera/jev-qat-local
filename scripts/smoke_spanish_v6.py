"""Check the public v6 helper and whichever default the activation gate chose."""
from pathlib import Path

from jev_qat import classify_spanish_message
from jev_qat.config import ROOT,save_json
from jev_qat.focus_experiment import read_json
from jev_qat.general_rescue_data import DIRECTORY


def main():
    run=Path(read_json(DIRECTORY/"latest-run.json")["directory"])
    summary=read_json(run/"summary.json")
    results=[]
    for message,expected in (("Hola, ¿cómo estás?","general"),("Añade arroz a mi lista de compras","lists"),("Léeme los titulares del periódico de hoy","news")):
        result=classify_spanish_message(message,version="v6",calibrated=bool(summary["profile_name"]))
        assert result["category"]==expected
        assert result["metadata"]["task_profile"]=="massive_es_v6"
        assert result["metadata"]["calibrated"]==bool(summary["profile_name"])
        results.append({"message":message,"response":result})
    expected_default=read_json(ROOT/"classification/current.json")["version"]
    default=classify_spanish_message("Hola, ¿cómo estás?")
    assert default["category"]=="general" and default["metadata"]["calibrated"]
    assert default["metadata"]["task_profile"]==f"massive_es_{expected_default}"
    if not summary["activation"]["approved"]:
        assert expected_default=="v2"
    save_json(run/"helper-smoke.json",{"passed":True,"v6":results,"default_version":expected_default,"default":default})
    print(f"Public helper passed; default is {expected_default}.")


if __name__=="__main__":
    main()
