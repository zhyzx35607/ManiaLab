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
                           tempo=dict(bpm=210,offset=93))
    assert result.name=='generated.osu'
    assert 'in_context=[TIMING]' in seen and 'output_type=[MAP]' in seen
    ref=sections((output/'timing-context.osu').read_text(encoding='utf-8'))
    assert not any(line.strip() for line in ref.get('HitObjects', []))
    red=next(line for line in ref['TimingPoints'] if line.strip()).split(',')
    assert float(red[0])==93 and abs(float(red[1])-60000/210)<1e-7


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
