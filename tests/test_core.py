from collections import Counter
from dataclasses import replace
from pathlib import Path
import io
import random
import zipfile

import pytest

from manialab.arrange import assign_lanes, parameters, select_notes
from manialab.domain import Note, export_osu, invariant, parse_osu, validate
from manialab.planner import validate_plan
from manialab.storage import extract_osz, safe_path

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "vendor/mapperatorinator/template.osu"


@pytest.mark.parametrize("keys", range(6, 11))
def test_roundtrip(keys):
    original = [Note(1000, 3000, 0, id="a"), Note(1000, 4000, keys - 1, id="b"), Note(1500, 1500, 1, id="c")]
    text = export_osu(original, keys, TEMPLATE, title="中文曲名")
    parsed = parse_osu(text)
    assert parsed["keys"] == keys
    assert [(n.start, n.end, n.lane) for n in parsed["notes"]] == [(1000,3000,0),(1000,4000,keys-1),(1500,1500,1)]
    assert parsed["metadata"]["Title"] == "中文曲名"


@pytest.mark.parametrize("keys", range(6,11))
def test_remap_preserves_events(keys):
    notes = [Note(1000,5000,0,id="a"), Note(1000,4000,1,id="b"), Note(1500,1500,2,id="c"), Note(4000,6000,1,id="d")]
    original = export_osu(notes, keys, TEMPLATE).replace("1000,500", "1000,500")
    original = original.replace("[TimingPoints]\n", "[TimingPoints]\n0,500,4,2,0,70,1,0\n2500,400,4,2,0,70,1,0\n3000,-50,4,2,0,70,0,0\n")
    parsed = parse_osu(original)
    result = assign_lanes(parsed["notes"], {"keys":keys})
    assert invariant(result) == invariant(parsed["notes"])
    exported = parse_osu(export_osu(result, keys, TEMPLATE, original=original))
    assert exported["timing"] == parsed["timing"]
    assert Counter(tuple(n.raw.split(",")[1:]) for n in parsed["notes"]) == Counter(tuple(n.raw.split(",")[1:]) for n in exported["notes"])
    assert validate(result,keys)["valid"]


def test_capacity_and_zero_length():
    with pytest.raises(ValueError, match="1000ms"):
        assign_lanes([Note(0,5000,i,id=str(i)) for i in range(6)]+[Note(1000,1000,id="tap")],{"keys":6})
    assert not validate([Note(100,400,0),Note(200,200,0)],6)["valid"]
    assert not validate([Note(100,100,0),Note(100,100,0)],6)["valid"]
    with pytest.raises(ValueError):
        parse_osu(export_osu([Note(100,200,0)],6,TEMPLATE).replace("200:0:","100:0:"))


def test_locked_cross_boundary():
    locked=[Note(0,8000,0,id="fixed"),Note(4000,6000,1,id="future")]
    out=assign_lanes([Note(2000,5000,id="new")],{"keys":6},locked)
    assert next(n for n in out if n.id=="new").lane not in (0,1)
    assert [n for n in out if n.id=="fixed"]==[locked[0]]


def test_source_backed_ln():
    candidates=[Note(1000,5000,pitch=60,source="pitch",id="p"),Note(6000,6100,pitch=64,source="pitch",id="q"),Note(7000,7000,source="transient",id="t")]
    result,_=select_notes(candidates,{"ln_ratio":1})
    assert next(n for n in result if n.id=="p").end==5000
    assert all(n.end<=next(c.end for c in candidates if c.id==n.id) for n in result)
    taps,_=select_notes(candidates,{"ln_ratio":0})
    assert not any(n.hold for n in taps)


def test_random_interval_coloring():
    rng=random.Random(7)
    for keys in range(6,11):
        notes=[]
        for lane in range(keys):
            t=rng.randrange(100)
            for i in range(30):
                end=t+rng.randrange(0,700)
                notes.append(Note(t,end,lane,id=f"{lane}-{i}"))
                t=end+rng.randrange(1,300)
        result=assign_lanes(notes,{"keys":keys})
        assert invariant(result)==invariant(notes)
        assert validate(result,keys)["valid"]


@pytest.mark.parametrize("name",["../evil.osu","C:/evil.osu","/root.osu","a/../../b.osu","CON.osu"])
def test_unsafe_paths(tmp_path,name):
    with pytest.raises(ValueError):safe_path(tmp_path,name)


def test_zip_validation_before_extract(tmp_path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:
        z.writestr("good.osu","test")
        z.writestr("../bad.osu","test")
    buf.seek(0)
    with pytest.raises(ValueError):extract_osz(buf,tmp_path)
    assert not (tmp_path/"good.osu").exists()


def test_plan_bounded_and_no_times():
    segments=[{"start_ms":0,"end_ms":15000}]
    valid=validate_plan({"preferences":{"ln_ratio":.8},"segments":[{"index":0,"preferences":{"max_held":3}}]},segments)
    assert valid["segments"][0]["end_ms"]==15000
    for value in [{"preferences":{"keys":18}},{"preferences":{"density":float('nan')}},{"segments":[{"index":0,"start_ms":7,"preferences":{}}]}]:
        with pytest.raises(ValueError):validate_plan(value,segments)
