from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from scripts import piano_adaptation_data as data


def note(onset,pitch=60):
    return dict(onset=onset,pitch=pitch)


def test_target_uses_physical_subframes_including_the_second_residual():
    target = data.onset_targets([note(.103),note(.197)],0,frames=31)
    expected = np.maximum.reduce([np.maximum(0,1-abs(np.arange(31)/100-onset)/.05)
                                  for onset in (.103,.197)])
    assert target.dtype == np.float32 and target.shape == (31,88)
    np.testing.assert_allclose(target[:,39],expected,atol=3e-8,rtol=0)
    assert target[18,39] == pytest.approx(.66)
    assert not target[:,:39].any() and not target[:,40:].any()


def test_repeated_and_same_frame_onsets_take_max_without_overwriting():
    notes = [note(.101),note(.104),note(.117)]
    target = data.onset_targets(notes,0,frames=25)
    assert target[10,39] == pytest.approx(.98)
    assert target[12,39] == pytest.approx(.94)
    assert np.array_equal(target,data.onset_targets(list(reversed(notes)),0,frames=25))


def test_window_external_onsets_contribute_without_moving_the_clock():
    target = data.onset_targets([note(.98,21),note(1.12,108),note(.8,30),note(1.3,31)],1,frames=11)
    assert target[0,0] == pytest.approx(.6)
    assert target[-1,87] == pytest.approx(.6)
    assert not target[:,9:11].any()
    assert data.onset_targets([],1).sum() == 0


@pytest.mark.parametrize('notes',[[note(0,20)],[note(0,109)],[note(0,60.)],
                                  [note(float('nan'))],[note(-.01)],[note(20,200)]])
def test_invalid_midi_is_rejected_even_when_outside_window(notes):
    with pytest.raises(ValueError):
        data.onset_targets(notes,0)


@pytest.mark.parametrize('kwargs',[dict(start=-1),dict(start=float('inf')),dict(start=0,frames=0),
                                   dict(start=0,fps=0),dict(start=0,frames=True)])
def test_invalid_frame_clock_is_rejected(kwargs):
    with pytest.raises(ValueError):
        data.onset_targets([],**kwargs)


def test_window_ownership_is_exactly_once_for_every_physical_frame():
    windows = np.arange(1001)[None,:]+np.arange(35)[:,None]*500
    stitched = data.stitch(windows)
    assert data.window_slices()[0] == (0,750)
    assert data.window_slices()[-1] == (250,1000)
    assert sum(right-left for left,right in data.window_slices()) == 18000
    np.testing.assert_array_equal(stitched,np.arange(18000))
    assert len(stitched[:17000]) == 17000


def test_stitch_preserves_trailing_dimensions_and_handles_dict_and_lists():
    clock = np.arange(1001,dtype=np.float32)[None,:]+np.arange(3,dtype=np.float32)[:,None]*500
    windows = np.stack((clock,-clock),axis=-1)
    result = data.stitch(dict(onset=list(windows),pedal=clock[...,None]))
    assert result['onset'].shape == (2000,2)
    assert result['onset'].dtype == np.float32
    np.testing.assert_array_equal(result['onset'][:,0],np.arange(2000))
    np.testing.assert_array_equal(result['pedal'][:,0],np.arange(2000))
    assert data.stitch(np.zeros((1,1001,88))).shape == (1000,88)


@pytest.mark.parametrize('value',[np.zeros((0,1001,88)),np.zeros((2,1000,88)),{},
                                 dict(a=np.zeros((2,1001)),b=np.zeros((3,1001)))])
def test_invalid_window_collections_are_rejected(value):
    with pytest.raises(ValueError):
        data.stitch(value)


def test_pcm_keeps_scaling_and_averages_stereo_before_resampling(tmp_path):
    path = tmp_path/'stereo.wav'
    times = np.arange(44100)/44100
    left = np.round(4096*np.sin(2*np.pi*440*times)).astype(np.int16)
    right = np.round(2048*np.sin(2*np.pi*660*times)).astype(np.int16)
    raw = np.column_stack((left,right))
    wavfile.write(path,44100,raw)
    expected = resample_poly(raw.astype(np.float32).mean(axis=1)/32768,160,441).astype(np.float32)
    actual = data.read_pcm(path,seconds=1)
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape == (16000,) and actual.dtype == np.float32
    assert max(abs(actual)) < .1


def test_pcm_silence_from_antiphase_is_not_amplified(tmp_path):
    path = tmp_path/'antiphase.wav'
    signal = np.full(441,dtype=np.int16,fill_value=4096)
    wavfile.write(path,44100,np.column_stack((signal,-signal)))
    assert not data.read_pcm(path,seconds=.01).any()


@pytest.mark.parametrize('rate,raw,seconds',[(16000,np.zeros((160,2),np.int16),.01),
    (44100,np.zeros(441,np.int16),.01),(44100,np.zeros((441,2),np.float32),.01),
    (44100,np.zeros((441,2),np.int16),1)])
def test_pcm_rejects_wrong_original_format_or_short_input(tmp_path,rate,raw,seconds):
    path = tmp_path/'invalid.wav'
    wavfile.write(path,rate,raw)
    with pytest.raises(ValueError):
        data.read_pcm(path,seconds=seconds)


def make_roster(tmp_path,monkeypatch):
    rows,files = [],[]
    for index in range(10):
        split = 'train' if index<6 else 'validation'
        recording = dict(id=str(index),split=split,year=2004+index,members={})
        for kind in ('audio','midi'):
            path = tmp_path/'development'/split/(str(index)+'.'+kind)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((str(index)+' '+kind).encode())
            member = dict(name=f'archive/{index}.{kind}',size=path.stat().st_size)
            recording['members'][kind] = member
            files.append(dict(id=str(index),split=split,kind=kind,path=str(path),member=member,
                              size=member['size'],sha256=data._sha(path),zip_crc_verified=True))
        rows.append(recording)
    # A test row advertises no files and must never reach the file roster.
    rows.append(dict(id='reserved',split='test',audio='DO_NOT_READ',midi='DO_NOT_READ'))
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(rows=rows)),encoding='utf-8')
    digest = data._sha(selection)
    monkeypatch.setattr(data,'SELECTION_SHA256',digest)
    report = dict(selection_sha256=digest,files=files)
    report_path = tmp_path/'download-report.json'
    report_path.write_text(json.dumps(report),encoding='utf-8')
    class Header:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def getframerate(self):return 44100
        def getnchannels(self):return 2
        def getsampwidth(self):return 2
        def getcomptype(self):return 'NONE'
        def getnframes(self):return 180*44100
    monkeypatch.setattr(data.wave,'open',lambda *args:Header())
    monkeypatch.setattr(data,'_midi_labels',lambda path,duration:([note(.103)],[],[],{}))
    return report_path,report


def test_loader_hashes_all_development_and_preserves_roster_order(tmp_path,monkeypatch):
    _,report = make_roster(tmp_path,monkeypatch)
    seen = []
    original_sha = data._sha
    def checked(path):
        seen.append(Path(path))
        return original_sha(path)
    monkeypatch.setattr(data,'_sha',checked)
    rows = data.load_development(tmp_path)
    assert [row['id'] for row in rows] == [str(index) for index in range(10)]
    assert {Path(row['path']) for row in report['files']} <= set(seen)
    assert all(row['audio'].is_absolute() and row['midi'].is_absolute() for row in rows)
    assert rows[0]['notes'][0]['onset'] == .103
    assert not any('reserved' in str(path) or 'DO_NOT_READ' in str(path) for path in seen)


def test_test_download_entry_rejected_before_any_advertised_file_access(tmp_path,monkeypatch):
    report_path,report = make_roster(tmp_path,monkeypatch)
    report['files'][-1].update(split='test',path=str(tmp_path/'DO_NOT_READ'))
    report_path.write_text(json.dumps(report),encoding='utf-8')
    original_sha = data._sha
    seen = []
    def selection_only(path):
        seen.append(Path(path))
        assert Path(path).name == 'selection.json'
        return original_sha(path)
    monkeypatch.setattr(data,'_sha',selection_only)
    with pytest.raises(ValueError,match='Only selected development'):
        data.load_development(tmp_path)
    assert len(seen) == 1


def test_loader_stops_on_source_or_selection_identity_change(tmp_path,monkeypatch):
    _,report = make_roster(tmp_path,monkeypatch)
    Path(report['files'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='identity changed'):
        data.load_development(tmp_path)
    monkeypatch.setattr(data,'SELECTION_SHA256','0'*64)
    with pytest.raises(ValueError,match='selection changed'):
        data.load_development(tmp_path)


def test_loader_rejects_duplicate_or_escape_without_reading_target(tmp_path,monkeypatch):
    report_path,report = make_roster(tmp_path,monkeypatch)
    report['files'][-1] = deepcopy(report['files'][0])
    report_path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate development file'):
        data.load_development(tmp_path)
    report['files'][0]['path'] = str(tmp_path.parent/'outside-file')
    report_path.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='escaped'):
        data.load_development(tmp_path)
