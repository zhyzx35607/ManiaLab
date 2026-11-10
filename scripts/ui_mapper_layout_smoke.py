"""Exercise model-layout/conflict UI with a synthetic model, no weight download."""
import io
from pathlib import Path
import subprocess
import sys
import threading
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import Note,export_osu
from manialab.server import create_app
from manialab.upstream import MapperEngine
from werkzeug.serving import make_server


def main():
    root=Path(__file__).resolve().parents[1];folder=root/'tests/artifacts/model-layout-ui';folder.mkdir(parents=True,exist_ok=True)
    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        text=export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,root/'vendor/mapperatorinator/template.osu')
        text=text.replace('76,192,2000,1,','25,192,2000,1,')
        output.mkdir();target=output/'synthetic.osu';target.write_text(text,encoding='utf-8')
        return target
    MapperEngine.generate=model
    app=create_app(root,folder);client=app.test_client()
    wav=io.BytesIO()
    with wave.open(wav,'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(bytes(6*16000*2))
    raw=wav.getvalue()
    state=client.post('/api/import',data={'files':(io.BytesIO(raw),'fixture.wav')},headers={'X-ManiaLab-Token':app.config['TOKEN']}).json
    (folder/'projects'/state['id']/'preview.wav').write_bytes(raw)
    server=make_server('127.0.0.1',0,app,threaded=True);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=subprocess.run(['node',str(root/'scripts/ui_mapper_layout_smoke.cjs'),f'http://127.0.0.1:{server.server_port}',state['id']],check=True,capture_output=True,text=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    finally:
        server.shutdown();thread.join();server.server_close();app.extensions['executor'].shutdown()
    print(result.stdout,flush=True)


if __name__=='__main__':main()
