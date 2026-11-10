"""Exercise explicit instrument modes in the actual UI with symbolic evidence."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression
from manialab.server import create_app
from manialab.structure import edit_boundaries
from werkzeug.serving import make_server


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();folder=args.output.resolve()
    if folder.exists() or not folder.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local test directory')
    folder.mkdir(parents=True)
    app=create_app(ROOT,folder);client=app.test_client()
    wav=io.BytesIO()
    with wave.open(wav,'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(bytes(12*16000*2))
    raw=wav.getvalue()
    response=client.post('/api/import',data={'files':(io.BytesIO(raw),'instrument-controls.wav')},
                         headers={'X-ManiaLab-Token':app.config['TOKEN']})
    if response.status_code!=200:raise ValueError(response.json)
    state=response.json
    (folder/'projects'/state['id']/'preview.wav').write_bytes(raw)
    notes=[Note(1000,2500,pitch=60,source='pitch',voice='other',id='strings',instrument_id='yourmt3:program:48'),
           Note(1000,1500,pitch=64,source='pitch',voice='other',id='guitar',instrument_id='yourmt3:program:24'),
           Note(3000,3500,pitch=67,source='pitch',voice='other',id='unknown'),
           Note(8000,9500,pitch=60,source='pitch',voice='other',id='later',instrument_id='yourmt3:program:48')]
    analysis=dict(revision=8,duration=12,candidates=[n.data() for n in notes],phrase_candidates=[],waveform=[],
                  tempo=dict(bpm=120,offset=0,stable=True,division=4))
    analysis['structure']=edit_boundaries({},[0,6000,12000],analysis)
    analysis['segments']=analysis['structure']['segments']
    state.update(analysis=analysis,settings=parameters(dict(density=12,ln_ratio=1,chord_size=3)))
    state['expression']=propose_expression(analysis,state['settings'])
    state['expression']['source']='manual'
    for section in state['expression']['sections']:
        section['voices']['other']['mode']='onsets'
        # Legacy saved plans with labelled evidence should expose the controls
        # without regenerating and losing their existing section choices.
        section.pop('instrument_modes',None)
    app.extensions['projects'].save(state)
    server=make_server('127.0.0.1',0,app,threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=subprocess.run(['node',str(ROOT/'scripts/ui_instrument_modes_smoke.cjs'),
            f'http://127.0.0.1:{server.server_port}',state['id'],str(folder)],
            capture_output=True,text=True,encoding='utf-8',timeout=150,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (folder/'browser.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise ValueError(result.stdout+'\n'+result.stderr)
        print(result.stdout,flush=True)
    finally:
        server.shutdown();thread.join();server.server_close();app.extensions['executor'].shutdown()


if __name__=='__main__':main()
