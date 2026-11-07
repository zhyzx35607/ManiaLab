'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="manialab-token"]').content;
const numbers = ['density','ln_ratio','min_ln','max_held','release_gap','chord_size','pitch_follow','balance','jack_avoid','seed','onset_threshold','frame_threshold'];
let state = null, keys = 10, view = 'chart', busy = false, jobId = null, displayed = [], hitboxes = [], selected = null, manualTime = 0, audioContext = null, lastAudioTime = 0;
const audio = $('audio');
const msg = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error',error); };
async function api(path, data, form=false) {
  const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'X-ManiaLab-Token':token,...(form?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:form?data:JSON.stringify(data)});
  const result=await response.json(); if(!response.ok)throw Error(result.error||'请求失败'); return result;
}
const guarded=fn=>async(...args)=>{try{await fn(...args)}catch(e){msg(e.message,true)}};
function settings(){const p={keys,focus:$('focus').value};numbers.forEach(k=>p[k]=Number($(k).value));return p;}
function updateOutputs(){numbers.forEach(k=>{if($(k+'Out'))$(k+'Out').textContent=$(k).value;});}
function setSettings(p){keys=p.keys;numbers.forEach(k=>$(k).value=p[k]);$('focus').value=p.focus;updateOutputs();updateKeys();}
function updateKeys(){document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',Number(b.dataset.key)===keys));$('keytag').replaceChildren(document.createTextNode(keys));const s=document.createElement('span');s.textContent='KEYS';$('keytag').append(s);draw();}
function options(el,items){el.replaceChildren(...items.map(([value,text])=>{const o=document.createElement('option');o.value=value;o.textContent=text;return o;}));}
function variant(){return state&&state.active>=0?state.variants[state.active]:null;}
function duration(){return Math.max(state?.analysis?.duration||0,Number.isFinite(audio.duration)?audio.duration:0,displayed.reduce((m,n)=>Math.max(m,n.end/1000),0),(state?.input_notes||[]).reduce((m,n)=>Math.max(m,n.end/1000),0));}
function currentTime(){return audio.src&&Number.isFinite(audio.currentTime)?audio.currentTime:manualTime;}
function loadState(s){
  const changed=state?.id!==s.id;state=s;setSettings(s.settings);$('projectTitle').textContent=s.title;$('projectSub').textContent=s.mode==='remap'?`已导入 ${s.input_keys}K 原谱 · ${s.input_notes.length} 个物件`:'本地音频项目 · 先检查采音，再调整编排';
  $('mode').value=s.mode;options($('charts'),s.charts.map(c=>[c.path,c.label]));$('charts').classList.toggle('hidden',!s.charts.length);if(s.original)$('charts').value=s.original;
  if(s.audio){const src=`/api/projects/${s.id}/audio`;if(changed||!audio.src){audio.src=src;manualTime=0;}}else{audio.removeAttribute('src');audio.load();}
  const v=variant();if(v)$('mode').value=v.mode;
  const tempo=v||s.analysis?.tempo;if(tempo){$('bpm').value=tempo.bpm;$('offset').value=tempo.offset;}
  if(s.analysis)$('tempoNote').textContent=s.analysis.tempo.reliability+'。未自动吸附音符；重定轨仍使用原始变速。';
  $('planText').textContent=s.plan?.explanation||'';
  options($('variants'),s.variants.length?s.variants.map((v,i)=>[i,`候选 ${i+1} · ${v.settings.keys}K · seed ${v.settings.seed} · ${v.notes.length} 物件${v.region?' · 局部':''}`]):[['','尚未生成']]);$('variants').value=s.active;
  [['exportOsu','osu'],['exportOsz','osz'],['exportJson','json']].forEach(([id,kind])=>{const a=$(id);a.classList.toggle('disabled',!v||(kind==='osz'&&!s.audio));a.title=kind==='osz'&&!s.audio?'请导入配套音频后导出谱包':'';if(v)a.href=`/api/projects/${s.id}/export/${kind}`;else a.removeAttribute('href');a.download='ManiaLab.'+kind;});
  const report=v?.report;const vals=report?[report.notes,`${(report.ln_ratio*100).toFixed(1)}%`,report.max_held,`${(report.longest_ln_ms/1000).toFixed(2)}s`]:['—','—','—','—'];document.querySelectorAll('.metrics strong').forEach((e,i)=>e.textContent=vals[i]);
  selected=null;setMode();draw();refreshRecent();
}
async function refreshRecent(){const status=await api('/api/status');options($('recent'),[['','打开已保存项目…'],...status.projects.map(p=>[p.id,p.title])]);if(state)$('recent').value=state.id;$('engineStatus').textContent=status.engine.installed?`运行环境已就绪 · ${status.engine.model_downloaded?'模型权重已下载':'模型权重将在首次生成时下载'}`:'原模型未安装 · 不影响基础采音与定轨';}
function setMode(){const remap=$('mode').value==='remap';$('musical').classList.toggle('hidden',remap);$('strictHint').classList.toggle('hidden',!remap);$('bpm').disabled=remap;$('offset').disabled=remap;$('generate').firstChild.textContent=remap?'严格重新定轨 ':'生成候选谱面 ';}
function setBusy(value){busy=value;['analyze','generate','another','regenerate','plan','charts','variants','recent','files','clearPlan','engineInstall','engineGenerate'].forEach(id=>$(id).disabled=value);$('job').classList.toggle('hidden',!value);}
async function runJob(action,payload){
  if(!state)throw Error('请先导入素材');if(busy)return;setBusy(true);
  try{const job=await api(`/api/projects/${state.id}/${action}`,payload);jobId=job.id;
    while(true){const j=await api('/api/jobs/'+jobId);$('progress').value=j.progress;$('jobText').textContent=j.message;
      if(['done','error','cancelled'].includes(j.status)){if(j.status==='error')throw Error(j.message);if(j.status==='done'){loadState(await api('/api/projects/'+state.id));const v=variant();msg(action==='analyze'?`已提取 ${state.analysis.candidates.length} 个采音候选。切换「采音证据」试听检查。`:action==='plan'?'编排建议已应用，点击生成即可试谱。':action==='engine-install'?'原模型运行环境已安装。首次生成将自动下载 mania 权重。':`${v.report.notes} 个物件，${v.report.ln} 个 LN；结构检查通过。${v.selection.message||''}`);}else msg(j.message);break;}
      await new Promise(r=>setTimeout(r,550));
    }
  }finally{jobId=null;setBusy(false);}
}
async function importFiles(files){if(!files.length||busy)return;setBusy(true);$('jobText').textContent='导入素材…';try{const data=new FormData();for(const f of files)data.append('files',f);loadState(await api('/api/import',data,true));msg(state.mode==='remap'?'原谱已导入，可直接严格重新定轨。':'音频已导入，点击分析音频开始。');}finally{setBusy(false);$('files').value='';}}
async function generate(region){await runJob('generate',{settings:settings(),mode:$('mode').value,bpm:Number($('bpm').value),offset:Number($('offset').value),...(region?{region}:{})});}
numbers.forEach(k=>$(k).addEventListener('input',updateOutputs));
document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{keys=Number(b.dataset.key);updateKeys();});
$('files').onchange=guarded(e=>importFiles(e.target.files));
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('drag');};$('drop').ondragleave=()=>$('drop').classList.remove('drag');$('drop').ondrop=guarded(e=>{e.preventDefault();$('drop').classList.remove('drag');return importFiles(e.dataTransfer.files);});
$('recent').onchange=guarded(async e=>{if(e.target.value)loadState(await api('/api/projects/'+e.target.value));});
$('charts').onchange=guarded(async e=>{audio.removeAttribute('src');loadState(await api(`/api/projects/${state.id}/chart`,{path:e.target.value}));});
$('variants').onchange=guarded(async e=>loadState(await api(`/api/projects/${state.id}/variant`,{index:Number(e.target.value)})));
$('analyze').onclick=guarded(()=>runJob('analyze',{settings:settings()}));$('generate').onclick=guarded(()=>generate());
$('another').onclick=guarded(()=>{$('seed').value=Math.floor(Math.random()*2147483647);return generate();});
$('regenerate').onclick=guarded(()=>generate([Number($('regionStart').value)*1000,Number($('regionEnd').value)*1000]));
$('cancel').onclick=guarded(()=>jobId?api('/api/jobs/'+jobId+'/cancel',{}):null);
$('mode').onchange=setMode;
$('plan').onclick=guarded(()=>runJob('plan',{endpoint:$('endpoint').value,model:$('model').value,key:$('apiKey').value,instruction:$('instruction').value,settings:settings()}));
$('clearPlan').onclick=guarded(async()=>{if(state)loadState(await api(`/api/projects/${state.id}/clear-plan`,{}));});
$('engineInstall').onclick=guarded(()=>runJob('engine-install',{}));
$('engineGenerate').onclick=guarded(()=>runJob('engine-generate',{settings:settings(),seconds:Number($('engineSeconds').value)}));
$('preset').onchange=()=>{const presets={balanced:{density:7,ln_ratio:.55,min_ln:240,max_held:5,chord_size:4,focus:'balanced'},sustain:{density:6,ln_ratio:.85,min_ln:300,max_held:4,chord_size:3,focus:'pitched'},rhythm:{density:8,ln_ratio:.25,min_ln:240,max_held:3,chord_size:4,focus:'transient'}};setSettings({...settings(),...presets[$('preset').value]});};
function setView(v){view=v;selected=null;$('chartView').classList.toggle('active',v==='chart');$('evidenceView').classList.toggle('active',v==='evidence');draw();}
$('chartView').onclick=()=>setView('chart');$('evidenceView').onclick=()=>setView('evidence');$('window').onchange=draw;
$('helpBtn').onclick=()=>$('help').showModal();$('closeHelp').onclick=()=>$('help').close();
function seek(t){manualTime=Math.max(0,Math.min(duration(),t));if(audio.src&&Number.isFinite(audio.duration))audio.currentTime=manualTime;lastAudioTime=manualTime;draw();}
$('seek').oninput=()=>seek(Number($('seek').value));$('waveform').onclick=e=>{const r=$('waveform').getBoundingClientRect();seek((e.clientX-r.left)/r.width*duration());};
function canvas(id){const el=$(id),r=el.getBoundingClientRect(),d=window.devicePixelRatio||1;if(el.width!==Math.round(r.width*d)||el.height!==Math.round(r.height*d)){el.width=Math.round(r.width*d);el.height=Math.round(r.height*d);}const c=el.getContext('2d');c.setTransform(d,0,0,d,0,0);return[c,r.width,r.height];}
function fmt(t){return `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(3).padStart(6,'0')}`;}
function draw(){
  const v=variant();displayed=view==='evidence'?(state?.analysis?.candidates||[]):(v?.notes||state?.input_notes||[]);const t=currentTime(),span=Number($('window').value);const total=duration();$('seek').max=Math.max(1,total);$('seek').value=t;$('clock').textContent=fmt(t);$('emptyState').classList.toggle('hidden',displayed.length>0);
  const [w,ww,wh]=canvas('waveform');w.clearRect(0,0,ww,wh);w.fillStyle='#344e61';const wave=state?.analysis?.waveform||[];wave.forEach((a,i)=>{const x=i/wave.length*ww;w.fillRect(x,wh/2-a*wh*.44,Math.max(1,ww/wave.length),Math.max(1,a*wh*.88));});w.fillStyle='#78e4c4';w.fillRect(total?t/total*ww:0,0,2,wh);
  const [c,width,height]=canvas('roll');c.clearRect(0,0,width,height);const k=view==='evidence'?88:(v?.settings.keys||state?.input_keys||keys);const left=view==='evidence'?28:36;const cw=(width-left-12)/k;const bottom=height-40;const scale=(height-75)/span;hitboxes=[];
  if(view==='evidence'){c.fillStyle='#627d94';c.font='10px Segoe UI';c.fillText('低音 ← 音高 → 高音',left+5,17);}
  for(let lane=0;lane<k;lane++){c.fillStyle=lane%2===0?'#142131':'#101b29';c.fillRect(left+lane*cw,25,cw-1,height-25);if(view==='chart'){c.fillStyle='#8199af';c.font='11px Segoe UI';c.textAlign='center';c.fillText(String(lane+1),left+(lane+.5)*cw,height-12);}}
  c.textAlign='left';c.font='10px Segoe UI';for(let second=Math.ceil(t);second<t+span;second++){const y=bottom-(second-t)*scale;c.strokeStyle='#27374b';c.beginPath();c.moveTo(left,y);c.lineTo(width-12,y);c.stroke();c.fillStyle='#718399';c.fillText(`${second}s`,3,y+3);}
  for(const n of displayed){if(n.end/1000<t-.2||n.start/1000>t+span)continue;const lane=view==='evidence'?(n.source==='transient'?2:n.pitch-21):n.lane;if(lane<0||lane>=k)continue;const x=left+lane*cw+Math.min(3,cw*.1),bw=Math.max(2,cw-Math.min(6,cw*.2));const y=bottom-(n.start/1000-t)*scale,ye=bottom-(n.end/1000-t)*scale;const hold=n.end>n.start;c.fillStyle=n.id===selected?'#ffffff':hold?'#78e4c4':'#ffb978';c.globalAlpha=hold?.8:.95;const top=Math.max(25,ye),end=Math.min(bottom+5,y);if(end>=25&&top<=bottom+5){c.fillRect(x,hold?top:end-4,bw,hold?Math.max(5,end-top):5);hitboxes.push({n,x,y:hold?top:end-6,w:bw,h:hold?Math.max(7,end-top):9});}c.globalAlpha=1;}
  c.strokeStyle='#d4f9ef';c.lineWidth=1.5;c.beginPath();c.moveTo(left,bottom);c.lineTo(width-12,bottom);c.stroke();
}
$('roll').onclick=e=>{const rect=$('roll').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=[...hitboxes].reverse().find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);if(hit){const n=hit.n;selected=n.id;$('inspector').textContent=`${n.end>n.start?'LN / 持续音':'短键 / 瞬态'} · 起点 ${n.start} ms · 终点 ${n.end} ms · 长度 ${n.end-n.start} ms · ${n.source==='import'?'原谱物件':n.source==='pitch'?`检测音高 MIDI ${n.pitch} · 激活强度 ${n.strength}`:'频谱瞬态'}${n.lane>=0?' · 轨道 '+(n.lane+1):''}`;draw();}};
audio.addEventListener('play',()=>{if(!audioContext&&$('clicks').checked)audioContext=new (window.AudioContext||window.webkitAudioContext)();audioContext?.resume();lastAudioTime=audio.currentTime;});
function tick(){const now=currentTime();if(!audio.paused){if($('clicks').checked&&audioContext&&now-lastAudioTime<.3){const notes=variant()?.notes||[];const attacks=notes.filter(n=>n.start/1000>lastAudioTime&&n.start/1000<=now);if(attacks.length){const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.value=850;g.gain.setValueAtTime(.04,audioContext.currentTime);g.gain.exponentialRampToValueAtTime(.001,audioContext.currentTime+.025);o.connect(g).connect(audioContext.destination);o.start();o.stop(audioContext.currentTime+.03);}}lastAudioTime=now;draw();}requestAnimationFrame(tick);}
new ResizeObserver(draw).observe($('roll'));audio.addEventListener('loadedmetadata',draw);audio.addEventListener('seeked',()=>{lastAudioTime=audio.currentTime;draw();});
updateOutputs();guarded(refreshRecent)();draw();requestAnimationFrame(tick);
