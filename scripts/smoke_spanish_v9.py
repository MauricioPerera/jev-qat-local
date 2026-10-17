"""Explicit v9 helper, scoped calibration and unchanged v2 default."""
from jev_qat.config import save_json
from jev_qat.focus_experiment import read_json
from jev_qat.spanish import classify_spanish_message
from jev_qat.stable_rescue_data import DIRECTORY


def main():
    rows=[]
    for text,target in (("hoy he tenido una reunión con jorge","general"),
                        ("Añade pan y leche a mi lista de compras","lists"),
                        ("Muéstrame las noticias de hoy","news")):
        r=classify_spanish_message(text,version="v9")
        assert r["category"]==target and r["metadata"]["calibrated"] and r["metadata"]["task_profile"]=="massive_es_v9"
        rows.append({"text":text,"expected":target,"result":r})
    current=read_json(DIRECTORY/"latest-run.json")
    from pathlib import Path
    run=Path(current["directory"])
    default=classify_spanish_message("Añade pan a mi lista de compras")
    assert default["category"]=="lists" and default["metadata"]["task_profile"]=="massive_es_v2"
    save_json(run/"helper-smoke.json",{"passed":True,"explicit_v9":rows,"default_v2":default})
    print("Explicit calibrated v9 general/lists/news and unchanged v2 default passed",flush=True)


if __name__=="__main__":
    main()
