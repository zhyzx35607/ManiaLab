from pathlib import Path
import subprocess
import os
import sys
import time

import pytest

from manialab.arrange import parameters
from manialab.domain import Note,export_osu,sections,parse_osu
from manialab.upstream import MapperEngine,fixed_timing_output,verify_fixed_timing

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('keys',range(6,11))
def test_generated_model_timing_is_enforced_after_serialization(keys):
    tempo=dict(bpm=210,offset=93,division=4,audio_shift_ms=22.5)
    original=export_osu([Note(378,950,0),Note(435,435,1),Note(490,500,2)],keys,
        ROOT/'vendor/mapperatorinator/template.osu',bpm=209.998,offset=91)
    # Preserve hit sounds and custom samples while rewriting the model clock.
    original=original.replace('950:0:0:0:0:', '950:2:3:60:release.wav')
    content,notes,report=fixed_timing_output(original,tempo)
    assert len(notes)==3 and report['collapsed_holds']==1
    assert [(n.start,n.end) for n in notes]==[(379,950),(450,450),(522,522)]
    assert [(n.evidence_start,n.evidence_end) for n in notes]==[(378,950),(435,435),(490,500)]
    assert notes[0].raw.endswith('950:2:3:60:release.wav')
    exported=export_osu(notes,keys,ROOT/'vendor/mapperatorinator/template.osu',original=content)
    assert verify_fixed_timing(exported,tempo)['off_grid']==0
    assert [(n.start,n.end) for n in parse_osu(exported)['notes']]==[(n.start,n.end) for n in notes]
    with pytest.raises(ValueError,match='落线'):
        verify_fixed_timing(exported.replace(',379,128,',',378,128,'),tempo)
    with pytest.raises(ValueError,match='红线'):
        verify_fixed_timing(exported,tempo|{'offset':91})


def test_confirmed_timing_context_is_passed_but_never_returned_as_generated_chart(tmp_path,monkeypatch):
    engine=MapperEngine(tmp_path,ROOT)
    monkeypatch.setattr(engine,'status',lambda:dict(installed=True,model_downloaded=True))
    monkeypatch.setattr('manialab.audio.run_hidden',lambda *a,**k:subprocess.CompletedProcess(a,0))
    output=tmp_path/'output';seen=[]
    def run(argv,*args):
        seen.extend(argv)
        text=export_osu([Note(1093,2093,0,id='model')],10,ROOT/'vendor/mapperatorinator/template.osu',bpm=210,offset=93)
        (output/'generated.osu').write_text(text,encoding='utf-8')
        return ''
    monkeypatch.setattr(engine,'run',run)
    result=engine.generate(tmp_path/'input.wav',output,parameters(),lambda *a:None,lambda:False,
                           tempo=dict(bpm=210,offset=93),seconds=12,start_ms=24000)
    assert result.name=='generated.osu'
    assert 'in_context=[TIMING]' in seen and 'output_type=[MAP]' in seen
    assert 'start_time=24000' in seen and 'end_time=36000' in seen
    ref=sections((output/'timing-context.osu').read_text(encoding='utf-8'))
    assert not any(line.strip() for line in ref.get('HitObjects', []))
    red=next(line for line in ref['TimingPoints'] if line.strip()).split(',')
    assert float(red[0])==93 and abs(float(red[1])-60000/210)<1e-7


def test_experimental_adapter_preserves_reference_meter_changes_without_notes(tmp_path,monkeypatch):
    engine=MapperEngine(tmp_path,ROOT)
    monkeypatch.setattr(engine,'status',lambda:dict(installed=True,model_downloaded=True))
    monkeypatch.setattr('manialab.audio.run_hidden',lambda *a,**k:subprocess.CompletedProcess(a,0))
    reference=tmp_path/'human.osu'
    original=export_osu([Note(1000,5000,1)],10,ROOT/'vendor/mapperatorinator/template.osu',bpm=70,offset=189)
    parts=sections(original)
    timing=['189,857.142857142857,3,2,1,30,1,0','45323,631.578947368421,4,2,1,30,1,1']
    original=original.replace('\n'.join(parts['TimingPoints']), '\n'.join(timing))
    reference.write_text(original,encoding='utf-8')
    adapter=tmp_path/'adapter';adapter.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):(adapter/name).touch()
    output=tmp_path/'generated';seen=[]
    def run(argv,*args):
        seen.extend(argv)
        (output/'result.osu').write_text(original,encoding='utf-8')
        return ''
    monkeypatch.setattr(engine,'run',run)
    result=engine.generate(tmp_path/'input.wav',output,parameters(),lambda *a:None,lambda:False,
                           timing_reference=reference,lora_path=adapter,difficulty=3.75,
                           conditioning={'add_year_token':False,'add_song_position_token':False})
    context=sections((output/'timing-context.osu').read_text(encoding='utf-8'))
    assert [x for x in context['TimingPoints'] if x.strip()]==timing
    assert not any(x.strip() for x in context.get('HitObjects', []))
    assert any(str(x).startswith('lora_path=') and adapter.as_posix() in str(x) for x in seen)
    assert 'difficulty=3.75' in seen
    assert 'train.data.add_year_token=false' in seen and 'train.data.add_song_position_token=false' in seen
    assert result.name=='result.osu'
    with pytest.raises(ValueError,match='不能同时'):
        engine.generate('',output,parameters(),lambda *a:None,lambda:False,tempo={},timing_reference=reference)


def test_arrangement_context_is_explicit_and_never_returned_as_prediction(tmp_path,monkeypatch):
    engine=MapperEngine(tmp_path,ROOT)
    monkeypatch.setattr(engine,'status',lambda:dict(installed=True,model_downloaded=True))
    monkeypatch.setattr('manialab.audio.run_hidden',lambda *a,**k:subprocess.CompletedProcess(a,0))
    original=export_osu([Note(1000,1000,4),Note(1500,2500,6)],10,
        ROOT/'vendor/mapperatorinator/template.osu',bookmarks=[1000,5000],creator='private mapper')
    reference=tmp_path/'sketch.osu';reference.write_text(original,encoding='utf-8')
    output=tmp_path/'out';seen=[]
    def run(argv,*args):
        seen.extend(argv)
        (output/'prediction.osu').write_text(original,encoding='utf-8')
        return 'ManiaLab sketch context active: 20 events; map-delimited-sketch-v1\nManiaLab research event constraints active: mania-event-state-v1'
    monkeypatch.setattr(engine,'run',run)
    result=engine.generate(tmp_path/'input.wav',output,parameters({'keys':10}),lambda *a:None,lambda:False,
        arrangement_reference=reference,sketch_codec='map-delimited-sketch-v1')
    assert result.name=='prediction.osu' and 'in_context=[TIMING,GD]' in seen
    assert any(str(v).startswith('train.data.context_types=') and '${context_type:gd}' in str(v) for v in seen)
    content=(output/'arrangement-context.osu').read_text(encoding='utf-8')
    assert [(n.start,n.end,n.lane) for n in parse_osu(content)['notes']]==[(1000,1000,4),(1500,2500,6)]
    assert 'Bookmarks: 1000,5000' not in content and 'private mapper' not in content
    assert '--mania-event-state=mania-event-state-v1' not in seen
    engine.generate(tmp_path/'input.wav',output,parameters({'keys':10}),lambda *a:None,lambda:False,
        arrangement_reference=reference,sketch_codec='map-delimited-sketch-v1',state_constraints=True)
    assert '--mania-event-state=mania-event-state-v1' in seen
    with pytest.raises(ValueError,match='实验事件约束'):
        engine.generate('',output,parameters(),lambda *a:None,lambda:False,state_constraints=True)
    with pytest.raises(ValueError,match='不原生支持'):
        engine.generate('',output,parameters(),lambda *a:None,lambda:False,arrangement_reference=reference)
    monkeypatch.setattr(engine,'run',lambda *a:'')
    with pytest.raises(ValueError,match='拒绝'):
        engine.generate('',output,parameters(),lambda *a:None,lambda:False,
            arrangement_reference=reference,sketch_codec='map-delimited-sketch-v1')
    with pytest.raises(ValueError,match='不能同时'):
        engine.generate('',output,parameters(),lambda *a:None,lambda:False,
            arrangement_reference=reference,timing_reference=reference)


@pytest.mark.skipif(os.name!='nt',reason='Windows venv process trees')
def test_cancelling_model_stops_its_child_process(tmp_path):
    import ctypes
    from ctypes import wintypes
    marker=tmp_path/'child.pid'
    executable=getattr(sys,'_base_executable',sys.executable)
    child="import time;time.sleep(120)"
    parent=("import subprocess,time;from pathlib import Path;"
        "p=subprocess.Popen(["+repr(executable)+",'-c',"+repr(child)+"]);"
        "marker=Path("+repr(str(marker))+");pending=marker.with_suffix('.pending');"
        "pending.write_text(str(p.pid));pending.replace(marker);time.sleep(120)")
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=None;deadline=time.monotonic()+20
    def cancel():
        nonlocal handle
        if marker.exists():
            handle=kernel.OpenProcess(0x00100001,False,int(marker.read_text()))
            assert handle and kernel.WaitForSingleObject(handle,0)==258
            return True
        if time.monotonic()>deadline:raise TimeoutError('Model child did not start')
        return False
    try:
        with pytest.raises(InterruptedError):
            MapperEngine(tmp_path,ROOT).run([executable,'-c',parent],'test',lambda *a:None,cancel)
        assert handle and kernel.WaitForSingleObject(handle,5000)==0,'Model child survived cancellation'
    finally:
        if handle:
            if kernel.WaitForSingleObject(handle,0)==258:kernel.TerminateProcess(handle,1)
            kernel.CloseHandle(handle)
