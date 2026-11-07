'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="manialab-token"]').content;
const numbers = ['density','ln_ratio','min_ln','max_held','release_gap','chord_size','pitch_follow','balance','jack_avoid','seed','onset_threshold','frame_threshold'];
let timingSections=[], lastScheduledBeat=-Infinity;
let state = null, keys = 10, view = 'chart', busy = false, jobId = null, displayed = [], hitboxes = [], selected = null, manualTime = 0, audioContext = null, lastAudioTime = 0;
const audio = $('audio');
const msg = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error',error); };
async function api(path, data, form=false) {
  const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'X-ManiaLab-Token':token,...(form?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:form?data:JSON.stringify(data)});
  const result=await response.json(); if(!response.ok)throw Error(result.error||'请求失败'); return result;
}
const guarded=fn=>async(...args)=>{try{await fn(...args)}catch(e){msg(e.message,true)}};
function settings(){const p={keys,focus:$('focus').value};numbers.forEach(k=>p[k]=Number($(k).value));return p;}
function updateOutputs(){numbers.forEach(k=>{if($(k+'Out'))$(k+'Out').textContent=k==='ln_ratio'?`${Math.round(Number($(k).value)*100)}%`:$(k).value;});}
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
  const tempo=v||s.analysis?.tempo;$('bpm').value=tempo?.bpm??120;$('offset').value=Math.round(tempo?.offset??0);
  $('timingConfirmed').checked=!!v?.timing_confirmed;$('division').value=tempo?.division||4;
  const tc=s.analysis?.tempo?.candidates||[];options($('tempoCandidates'),[['','选择其他节拍解释…'],...tc.map((c,i)=>[i,`${c.bpm} BPM · 偏移 ${c.offset} ms · 匹配 ${(c.coverage*100).toFixed(0)}%`])]);
  $('tempoNote').textContent=s.analysis?.tempo?.reliability||'分析后校准 BPM / Offset，再生成谱面。';
  $('timingReport').textContent=s.analysis?`起音融合 ${s.analysis.fused_attacks||0} 处 · 分析耗时 ${s.analysis.elapsed_seconds??'未知'} 秒 · ${s.analysis.cached?'复用音高缓存':'本次运行音高模型'}`:'';
  timingSections=v?.timing_sections||s.analysis?.tempo?.sections||[];
  $('planText').textContent=s.plan?.explanation||'';
  options($('variants'),s.variants.length?s.variants.map((v,i)=>[i,`候选 ${i+1} · ${v.settings.keys}K · seed ${v.settings.seed} · ${v.notes.length} 物件${v.region?' · 局部':''}`]):[['','尚未生成']]);$('variants').value=s.active;
  [['exportOsu','osu'],['exportOsz','osz'],['exportJson','json']].forEach(([id,kind])=>{const a=$(id);a.classList.toggle('disabled',!v||(kind==='osz'&&!s.audio));a.title=kind==='osz'&&!s.audio?'请导入配套音频后导出谱包':'';if(v)a.href=`/api/projects/${s.id}/export/${kind}`;else a.removeAttribute('href');a.download='ManiaLab.'+kind;});
  const report=v?.report;const vals=report?[report.notes,`${(report.ln_ratio*100).toFixed(1)}%`,report.max_held,`${(report.longest_ln_ms/1000).toFixed(2)}s`]:['—','—','—','—'];document.querySelectorAll('.metrics strong').forEach((e,i)=>e.textContent=vals[i]);
  selected=null;setMode();updateGridStatus();draw();refreshRecent();
}
function updateGridStatus(){
  const v=variant(),audioMode=v?.mode==='audio';
  const mismatch=audioMode&&(['bpm','offset','division'].some(k=>Number($(k).value)!==v[k]));
  const stale=audioMode&&(v.timing_revision||0)<3;
  const g=v?.report?.grid;
  const diagnosis=$('exportJson'),hasDiagnosis=!!(v||state?.analysis);
  diagnosis.classList.toggle('disabled',!hasDiagnosis);if(hasDiagnosis)diagnosis.href=`/api/projects/${state.id}/export/json`;else diagnosis.removeAttribute('href');
  $('gridReport').textContent=stale?'旧版候选：请重新分析并生成整谱，才能导出。':mismatch?'当前 BPM / Offset / 细分与预览候选不同，请重新生成后导出。':g?`落线检查：${g.endpoints-g.off_grid} / ${g.endpoints} 个短键起点与 LN 起止在网格上；最大舍入误差 ${g.max_error_ms} ms。落线检查不判断采音是否正确。`:'';
  ['exportOsu','exportOsz'].forEach(id=>{const a=$(id),blocked=!v||stale||mismatch||(id==='exportOsz'&&!state?.audio);a.classList.toggle('disabled',blocked);if(blocked)a.removeAttribute('href');else a.href=`/api/projects/${state.id}/export/${id==='exportOsu'?'osu':'osz'}`;});
}
async function refreshRecent(){const status=await api('/api/status');options($('recent'),[['','打开已保存项目…'],...status.projects.map(p=>[p.id,p.title])]);if(state)$('recent').value=state.id;$('engineStatus').textContent=status.engine.installed?`运行环境已就绪 · ${status.engine.model_downloaded?'模型权重已下载':'模型权重将在首次生成时下载'}`:'原模型未安装 · 不影响基础采音与定轨';}
function setMode(){const remap=$('mode').value==='remap';$('musical').classList.toggle('hidden',remap);$('strictHint').classList.toggle('hidden',!remap);$('bpm').disabled=remap;$('offset').disabled=remap;['tempoCandidates','halfBpm','doubleBpm','division','timingConfirmed','checkTiming','refitTiming','referenceTiming','applyTiming'].forEach(id=>$(id).disabled=remap);$('generate').firstChild.textContent=remap?'严格重新定轨 ':'生成候选谱面 ';}
function setBusy(value){busy=value;['analyze','generate','another','regenerate','plan','charts','variants','recent','files','clearPlan','engineInstall','engineGenerate'].forEach(id=>$(id).disabled=value);$('job').classList.toggle('hidden',!value);}
async function runJob(action,payload){
  if(!state)throw Error('请先导入素材');if(busy)return;setBusy(true);
  try{const job=await api(`/api/projects/${state.id}/${action}`,payload);jobId=job.id;
    while(true){const j=await api('/api/jobs/'+jobId);$('progress').value=j.progress;$('jobText').textContent=j.message;
      if(['done','error','cancelled'].includes(j.status)){if(j.status==='error')throw Error(j.message);if(j.status==='done'){loadState(await api('/api/projects/'+state.id));const v=variant();msg(action==='analyze'?`已提取 ${state.analysis.candidates.length} 个采音候选。切换「采音证据」试听检查。`:action==='plan'?'编排建议已应用，点击生成即可试谱。':action==='engine-install'?'原模型运行环境已安装。首次生成将自动下载 mania 权重。':`${v.report.notes} 个物件，${v.report.ln} 个 LN；结构检查通过。${v.selection.message||''}\n编排耗时 ${j.elapsed_seconds??v.elapsed_seconds??'未知'} 秒。${v.selection.feasible_ln_ratio!==undefined?` 可达 LN 比例${v.selection.maximum_proven?'上限':'（已求得）'} ${(v.selection.feasible_ln_ratio*100).toFixed(1)}%。`:''}`);}else msg(j.message);break;}
      await new Promise(r=>setTimeout(r,550));
    }
  }finally{jobId=null;setBusy(false);}
}
async function importFiles(files){if(!files.length||busy)return;setBusy(true);$('jobText').textContent='导入素材…';try{const data=new FormData();for(const f of files)data.append('files',f);loadState(await api('/api/import',data,true));msg(state.mode==='remap'?'原谱已导入，可直接严格重新定轨。':'音频已导入，点击分析音频开始。');}finally{setBusy(false);$('files').value='';}}
async function generate(region){$('offset').value=Math.round(Number($('offset').value));await runJob('generate',{settings:settings(),mode:$('mode').value,bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value),snap:true,timing_confirmed:$('timingConfirmed').checked,...(region?{region}:{})});}
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
  const [w,ww,wh]=canvas('waveform');w.clearRect(0,0,ww,wh);w.fillStyle='#344e61';const wave=state?.analysis?.waveform||[];wave.forEach((a,i)=>{const x=i/wave.length*ww;w.fillRect(x,wh/2-a*wh*.44,Math.max(1,ww/wave.length),Math.max(1,a*wh*.88));});for(const section of timingSections){if(!section.suspect||!total)continue;w.fillStyle='#dc795f';w.fillRect(section.start_ms/1000/total*ww,wh-4,(section.end_ms-section.start_ms)/1000/total*ww,4);}w.fillStyle='#78e4c4';w.fillRect(total?t/total*ww:0,0,2,wh);
  const [c,width,height]=canvas('roll');c.clearRect(0,0,width,height);const k=view==='evidence'?88:(v?.settings.keys||state?.input_keys||keys);const left=view==='evidence'?28:36;const cw=(width-left-12)/k;const bottom=height-40;const scale=(height-75)/span;hitboxes=[];
  if(view==='evidence'){c.fillStyle='#627d94';c.font='10px Segoe UI';c.fillText('低音 ← 音高 → 高音',left+5,17);}
  for(let lane=0;lane<k;lane++){c.fillStyle=lane%2===0?'#142131':'#101b29';c.fillRect(left+lane*cw,25,cw-1,height-25);if(view==='chart'){c.fillStyle='#8199af';c.font='11px Segoe UI';c.textAlign='center';c.fillText(String(lane+1),left+(lane+.5)*cw,height-12);}}
  c.textAlign='left';c.font='10px Segoe UI';for(let second=Math.ceil(t);second<t+span;second++){const y=bottom-(second-t)*scale;c.strokeStyle='#27374b';c.beginPath();c.moveTo(left,y);c.lineTo(width-12,y);c.stroke();c.fillStyle='#718399';c.fillText(`${second}s`,3,y+3);}
  drawBeatGrid(c,left,width,bottom,scale,t,span);
  for(const n of displayed){if(n.end/1000<t-.2||n.start/1000>t+span)continue;const lane=view==='evidence'?(n.source==='transient'?2:n.pitch-21):n.lane;if(lane<0||lane>=k)continue;const x=left+lane*cw+Math.min(3,cw*.1),bw=Math.max(2,cw-Math.min(6,cw*.2));const y=bottom-(n.start/1000-t)*scale,ye=bottom-(n.end/1000-t)*scale;const hold=n.end>n.start;c.fillStyle=n.id===selected?'#ffffff':hold?'#78e4c4':'#ffb978';c.globalAlpha=hold?.8:.95;const top=Math.max(25,ye),end=Math.min(bottom+5,y);if(end>=25&&top<=bottom+5){c.fillRect(x,hold?top:end-4,bw,hold?Math.max(5,end-top):5);hitboxes.push({n,x,y:hold?top:end-6,w:bw,h:hold?Math.max(7,end-top):9});}c.globalAlpha=1;}
  c.strokeStyle='#d4f9ef';c.lineWidth=1.5;c.beginPath();c.moveTo(left,bottom);c.lineTo(width-12,bottom);c.stroke();
}
$('roll').onclick=e=>{const rect=$('roll').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=[...hitboxes].reverse().find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);if(hit){const n=hit.n;selected=n.id;$('inspector').textContent=`${n.end>n.start?'LN / 持续音':'短键 / 瞬态'} · 起点 ${n.start} ms · 终点 ${n.end} ms · 长度 ${n.end-n.start} ms · ${n.source==='import'?'原谱物件':n.source==='pitch'?`检测音高 MIDI ${n.pitch} · 激活强度 ${n.strength}`:'频谱瞬态'}${n.lane>=0?' · 轨道 '+(n.lane+1):''}`;draw();}};
audio.addEventListener('play',()=>{if(!audioContext&&($('clicks').checked||$('metronome').checked))audioContext=new (window.AudioContext||window.webkitAudioContext)();audioContext?.resume();lastAudioTime=audio.currentTime;});
function tick(){const now=currentTime();if(!audio.paused){scheduleMetronome(now);if($('clicks').checked&&audioContext&&now-lastAudioTime<.3){const notes=variant()?.notes||[];const attacks=notes.filter(n=>n.start/1000>lastAudioTime&&n.start/1000<=now);if(attacks.length){const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.value=850;g.gain.setValueAtTime(.04,audioContext.currentTime);g.gain.exponentialRampToValueAtTime(.001,audioContext.currentTime+.025);o.connect(g).connect(audioContext.destination);o.start();o.stop(audioContext.currentTime+.03);}}lastAudioTime=now;draw();}requestAnimationFrame(tick);}
new ResizeObserver(draw).observe($('roll'));audio.addEventListener('loadedmetadata',draw);audio.addEventListener('seeked',()=>{lastAudioTime=audio.currentTime;draw();});
function timingPoints(){
  if($('mode').value==='remap')return (state?.input_timing||[]).map(line=>line.split(',').map(Number)).filter(p=>p.length>=7&&p[1]>0&&p[6]===1).map(p=>({offset:p[0],beat:p[1]})).sort((a,b)=>a.offset-b.offset);
  const bpm=Number($('bpm').value),offset=Number($('offset').value);
  return Number.isFinite(bpm)&&bpm>=10&&bpm<=1000&&Number.isFinite(offset)?[{offset,beat:60000/bpm}]:[];
}
function beatsBetween(start,end){
  const points=timingPoints(),out=[];
  points.forEach((p,i)=>{const finish=points[i+1]?.offset??Infinity;const a=i===0?start*1000:Math.max(start*1000,p.offset),b=Math.min(end*1000,finish);
    for(let j=Math.ceil((a-p.offset)/p.beat);p.offset+j*p.beat<b&&out.length<500;j++)out.push({time:(p.offset+j*p.beat)/1000,accent:j%4===0});});return out;
}
function drawBeatGrid(c,left,width,bottom,scale,t,span){
  if($('mode').value==='audio'){
    const p=timingPoints()[0],div=Number($('division').value);
    if(p){const tick=p.beat/div;for(let j=Math.ceil((t*1000-p.offset)/tick),count=0;p.offset+j*tick<(t+span)*1000&&count<1500;j++,count++){
      if(j%div===0)continue;const y=bottom-((p.offset+j*tick)/1000-t)*scale;c.strokeStyle=j%div===div/2?'#334952':'#21343d';c.lineWidth=.7;c.beginPath();c.moveTo(left,y);c.lineTo(width-12,y);c.stroke();
    }}
  }
  for(const b of beatsBetween(t,t+span)){const y=bottom-(b.time-t)*scale;c.strokeStyle=b.accent?'#52766d':'#2e4a4a';c.lineWidth=b.accent?1.5:1;c.beginPath();c.moveTo(left,y);c.lineTo(width-12,y);c.stroke();}
  c.lineWidth=1;
}
function timingChanged(){ $('timingConfirmed').checked=false;timingSections=[];lastScheduledBeat=-Infinity;updateGridStatus();draw(); }
['bpm','offset','division'].forEach(id=>$(id).addEventListener('input',timingChanged));
$('tempoCandidates').onchange=()=>{const i=$('tempoCandidates').value;if(i==='')return;const c=state.analysis.tempo.candidates[Number(i)];$('bpm').value=c.bpm;$('offset').value=c.offset;$('division').value=c.division;timingChanged();};
$('halfBpm').onclick=()=>{$('bpm').value=Math.max(10,Number($('bpm').value)/2);timingChanged();};
$('doubleBpm').onclick=()=>{$('bpm').value=Math.min(1000,Number($('bpm').value)*2);timingChanged();};
$('applyTiming').onclick=guarded(()=>{
  const text=$('referenceTiming').value;
  const bpm=text.match(/BPM\s*[:：=]\s*([\d.]+)/i),offset=text.match(/Offset\s*[:：=]\s*([+-]?[\d.]+)/i);
  const pair=text.trim().split(/[\s,，;；]+/).map(Number);
  const b=bpm?Number(bpm[1]):pair.length===2?pair[0]:NaN,o=offset?Number(offset[1]):pair.length===2?pair[1]:NaN;
  if(!Number.isFinite(b)||b<10||b>1000||!Number.isFinite(o)||Math.abs(o)>600000)throw Error('请粘贴 BPM: 210 与 Offset: 91，或两个用空格分隔的数字');
  $('bpm').value=b;$('offset').value=Math.round(o);timingChanged();msg(`已应用参考：${b} BPM，Offset ${Math.round(o)} ms。请试听确认后重新生成整谱。`);
});
$('refitTiming').onclick=guarded(async()=>{
  if(!state)throw Error('请先分析音频');
  const pid=state.id,bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value);
  const r=await api(`/api/projects/${pid}/timing-refit`,{bpm,offset,division});
  if(state.id!==pid||Number($('bpm').value)!==bpm||Number($('offset').value)!==offset||Number($('division').value)!==division)return;
  $('offset').value=r.offset;timingChanged();timingSections=r.sections;draw();
  $('timingReport').textContent=`按 ${r.bpm} BPM 重新估计 Offset：${r.offset} ms。${r.suspect} 段可能错位；请试听确认。`;
});
$('checkTiming').onclick=guarded(async()=>{if(!state)throw Error('请先分析音频');const r=await api(`/api/projects/${state.id}/timing-check`,{bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value)});timingSections=r.sections;const suspicious=r.sections.filter(s=>s.suspect);$('timingReport').textContent=`检查 ${r.sections.length} 段，${r.suspect} 段可能错位。`+(suspicious.length?'请定位试听：'+suspicious.slice(0,8).map(s=>`${(s.start_ms/1000).toFixed(0)}–${(s.end_ms/1000).toFixed(0)}s（相位偏差 ${s.phase_error_ms}ms）`).join('；'):'请再用节拍器确认倍速、半速和弱起。');draw();});
function scheduleMetronome(now){
  if(!$('metronome').checked||!audioContext)return;
  for(const b of beatsBetween(now,now+.12)){
    if(b.time<=lastScheduledBeat+.0001)continue;
    const at=audioContext.currentTime+(b.time-now)/audio.playbackRate;
    const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.value=b.accent?1500:1100;
    g.gain.setValueAtTime(.045,at);g.gain.exponentialRampToValueAtTime(.001,at+.025);o.connect(g).connect(audioContext.destination);o.start(at);o.stop(at+.03);lastScheduledBeat=b.time;
  }
}
['metronome','clicks'].forEach(id=>$(id).onchange=()=>{if(!audioContext)audioContext=new(window.AudioContext||window.webkitAudioContext)();audioContext.resume();lastScheduledBeat=-Infinity;});
audio.addEventListener('seeked',()=>{lastScheduledBeat=-Infinity;});audio.addEventListener('play',()=>{lastScheduledBeat=-Infinity;});
updateOutputs();guarded(refreshRecent)();draw();requestAnimationFrame(tick);
