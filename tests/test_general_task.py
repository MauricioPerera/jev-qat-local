from collections import Counter
import numpy as np
import pytest

from jev_qat.general_task import GeneralTask,select_examples


def fixtures():
    return [{"id":str(i),"target":label,"group_id":str(i)} for i,label in enumerate(
        ["qa","qa","qa","weather","calendar","email","lists","general","general","general"])]


def test_general_contrast_examples_are_added_without_repeating_ids():
    rows=fixtures()
    selected=select_examples(rows,np.linspace(1,.1,len(rows)),"unseen")
    assert len(selected)==8
    assert Counter(r["target"] for r in selected)["general"]==2
    assert len({r["id"] for r in selected})==len(selected)


def test_exact_query_match_is_excluded_from_both_retrieval_passes():
    rows=fixtures()
    selected=select_examples(rows,np.linspace(1,.1,len(rows)),"7")
    assert "7" not in {r["group_id"] for r in selected}
    assert Counter(r["target"] for r in selected)["general"]==2


def test_existing_general_neighbors_are_not_added_again():
    rows=fixtures()
    scores=np.array([.7,.6,.5,.4,.3,.2,.1,1,.9,.8])
    selected=select_examples(rows,scores,"unseen")
    assert len(selected)==6
    assert Counter(r["target"] for r in selected)["general"]==2


def test_general_task_rejects_incompatible_scope_before_loading():
    with pytest.raises(ValueError):
        GeneralTask().prepare("massive_es_v5",state="hola",question="otra pregunta",options=["sí","no"],modality="text")
