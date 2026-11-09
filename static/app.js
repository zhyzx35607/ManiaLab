'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="manialab-token"]').content;
const numbers = ['density','ln_ratio','min_ln','max_held','release_gap','chord_size','pitch_follow','balance','jack_avoid','seed','onset_threshold','frame_threshold','phrase_ln','pattern_prior','drums_weight','bass_weight','vocals_weight','other_weight','guitar_weight','piano_weight','ln_coordination','ln_release','ln_inverse'];
let timingSections=[], tempoChoices=[], lastScheduledBeat=-Infinity,separatorAvailable=false;
let lastClockShift=22.5;
let clockAnchor=22.5;
const clockShift=()=>Number($('audioShift').value);
const timingLocations=document.createElement('div');timingLocations.id='timingLocations';$('timingReport').after(timingLocations);
let state = null, keys = 10, view = 'chart', busy = false, jobId = null, displayed = [], hitboxes = [], selected = null, manualTime = 0, audioContext = null, lastAudioTime = 0;
const audio = $('audio');
const msg = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error',error); };
async function api(path, data, form=false) {
  const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'X-ManiaLab-Token':token,...(form?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:form?data:JSON.stringify(data)});
  const result=await response.json(); if(!response.ok)throw Error(result.error||'请求失败'); return result;
}
const guarded=fn=>async(...args)=>{try{await fn(...args)}catch(e){msg(e.message,true)}};
function settings(){const p={keys,focus:$('focus').value,layout_engine:$('layoutEngine').value};numbers.forEach(k=>p[k]=Number($(k).value));return p;}
function updateOutputs(){numbers.forEach(k=>{if($(k+'Out'))$(k+'Out').textContent=['ln_ratio','phrase_ln','pattern_prior','ln_coordination','ln_release','ln_inverse'].includes(k)?`${Math.round(Number($(k).value)*100)}%`:$(k).value;});}
function setSettings(p){keys=p.keys;numbers.forEach(k=>{$(k).value=p[k]??$(k).defaultValue;});$('layoutEngine').value=p.layout_engine||'beam';$('focus').value=p.focus||'balanced';updateOutputs();updateKeys();}
function updateKeys(){document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',Number(b.dataset.key)===keys));$('keytag').replaceChildren(document.createTextNode(keys));const s=document.createElement('span');s.textContent='KEYS';$('keytag').append(s);draw();}
function options(el,items){el.replaceChildren(...items.map(([value,text])=>{const o=document.createElement('option');o.value=value;o.textContent=text;return o;}));}
function showTempoChoices(){options($('tempoCandidates'),[['','选择其他节拍解释…'],...tempoChoices.map((c,i)=>[i,`${c.precision_suggestion?'精确候选 · ':''}${c.bpm} BPM · 偏移 ${c.offset} ms · 匹配 ${(c.coverage*100).toFixed(0)}%${c.precision_suggestion?` · 取整漂移约 ${c.rounding_drift_ms}ms`:''}`])]);}
function variant(){return state&&state.active>=0?state.variants[state.active]:null;}
function renderReview(){
  const v=variant(),body=$('reviewVoices'),events=$('reviewEvents');body.replaceChildren();events.replaceChildren();
  if(!v){$('reviewSummary').textContent='生成后显示声部取舍、长条和需要回听的位置。';return;}
  const notes=v.notes,names={mix:'混音 / 原谱',drums:'鼓 / 打击',bass:'贝斯',vocals:'人声',other:'其余伴奏',guitar:'吉他',piano:'钢琴'};
  for(const voice of [...new Set(notes.map(n=>n.voice||'mix'))]){
    const group=notes.filter(n=>(n.voice||'mix')===voice),holds=group.filter(n=>n.end>n.start),row=document.createElement('tr');
    for(const value of [names[voice]||voice,group.length,holds.length,Math.max(0,...holds.map(n=>n.end-n.start))+' ms']){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}body.append(row);
  }
  const shifted=notes.filter(n=>Number.isFinite(n.evidence_start)&&Math.abs(n.start-n.evidence_start)>30).sort((a,b)=>Math.abs(b.start-b.evidence_start)-Math.abs(a.start-a.evidence_start));
  const phrase=notes.filter(n=>n.phrase&&n.end>n.start).length;
  $('reviewSummary').textContent=v.mode==='remap'?'严格保时：原谱起止时间和 LN 类型保留。手感与声部表达仍需试听。':`保留 ${phrase} 段跨音高人声长条；${shifted.length} 个起音与原始检测相差超过 30 ms（含量化和手工修正）。点击下方位置回听；这些是检查线索，不是质量评分。`;
  for(const n of shifted.slice(0,12)){
    const button=document.createElement('button');button.className='quiet';button.textContent=`${fmt(n.start/1000)} · ${n.start-n.evidence_start>0?'+':''}${n.start-n.evidence_start} ms`;
    button.onclick=guarded(async()=>{seek(Math.max(0,n.start/1000-.3));setView('chart');inspectNote(n);if(window.loadWaveDetail)await loadWaveDetail(n.start);});events.append(button);
  }
  const review=v.report.review;
  if(review){
    const c=review.counts||{};$('reviewSummary').textContent+=` 手感检查：${c.short_release||0} 次短放键接续、${c.fast_repeat||0} 次快速同轨、${c.held_hand_load||0} 处同手 LN 配合密集。它们可能是有意设计，请结合目标难度试打。`;
    for(const item of (review.locations||[]).slice(0,16)){
      const button=document.createElement('button');button.className='quiet';button.textContent=`${fmt(item.start_ms/1000)} · ${item.text}`;
      button.onclick=()=>{setView('chart');seek(Math.max(0,item.start_ms/1000-.4));const n=notes.find(n=>n.start===item.start_ms&&(item.lane===null||n.lane===item.lane));if(n)inspectNote(n);};events.append(button);
    }
  }
}
function duration(){return Math.max(state?.analysis?.duration||0,Number.isFinite(audio.duration)?audio.duration:0,displayed.reduce((m,n)=>Math.max(m,n.end/1000),0),(state?.input_notes||[]).reduce((m,n)=>Math.max(m,n.end/1000),0));}
function currentTime(){return audio.src&&Number.isFinite(audio.currentTime)?audio.currentTime:manualTime;}
function renderTimingLocations(){
  timingLocations.replaceChildren();if(!state?.analysis)return;
  const length=state.analysis.duration||0,suspect=timingSections.filter(s=>s.suspect);
  const explanation=document.createElement('p');explanation.className='muted compact';explanation.textContent=suspect.length?'点击可疑区间，在对音实验室试听检查：':'建议分别试听开头、中间和结尾：';timingLocations.append(explanation);
  const locations=suspect.length?suspect.slice(0,12).map(s=>({time:s.start_ms/1000,label:`${(s.start_ms/1000).toFixed(0)}–${Math.min(length,s.end_ms/1000).toFixed(0)}s`,title:`${s.fallback?'鼓点不足，使用混音核对 · ':''}相位偏差 ${s.phase_error_ms}ms · 起音匹配 ${(s.coverage*100).toFixed(0)}%`})):[{time:0,label:'开头'},{time:Math.max(0,length/2-6),label:'中间'},{time:Math.max(0,length-12),label:'结尾'}];
  for(const item of locations){
    const button=document.createElement('button');button.className='quiet';button.textContent=item.label;button.title=item.title||'定位到对音实验室';
    button.onclick=()=>{const time=Math.min(Math.max(0,length-1),item.time);audio.pause();seek(time);$('auditionStart').value=time.toFixed(1);const panel=document.querySelector('.audition-panel');panel.open=true;$('auditionNote').textContent=`已定位 ${time.toFixed(1)} 秒；点击「试听当前拍线 A」比较音乐与拍线，或设置 B 的 Offset 做对照。`;panel.scrollIntoView({block:'center',behavior:'smooth'});};timingLocations.append(button);
  }
}
function loadState(s){
  const changed=state?.id!==s.id;auditionRequest++;cueClock.reset();$('auditionAudio').pause();if(auditionUrl){URL.revokeObjectURL(auditionUrl);auditionUrl=null;}$('auditionAudio').removeAttribute('src');state=s;setSettings(s.settings);$('projectTitle').textContent=s.title;$('projectSub').textContent=s.mode==='remap'?`已导入 ${s.input_keys}K 原谱 · ${s.input_notes.length} 个物件`:'本地音频项目 · 先检查采音，再调整编排';
  $('mode').value=s.mode;options($('charts'),s.charts.map(c=>[c.path,c.label]));$('charts').classList.toggle('hidden',!s.charts.length);if(s.original)$('charts').value=s.original;
  if(s.audio){const src=`/api/projects/${s.id}/audio?revision=${s.analysis?.revision||0}&elapsed=${s.analysis?.elapsed_seconds||0}`;if(changed||!audio.src||!audio.src.endsWith(src)){audio.src=src;manualTime=0;}}else{audio.removeAttribute('src');audio.load();}
  const v=variant();if(v)$('mode').value=v.mode;if(s.analysis)$('analysisEngine').value=s.analysis.separation?(s.analysis.separation.source==='provided'?'provided':s.analysis.separation.profile==='six'?'deep6':'deep'):'basic';options($('voice'),[['mix','全部声部'],...(s.analysis?.voices||[]).map(v=>[v.voice,({drums:'鼓 / 打击',bass:'贝斯',vocals:'人声',other:'其余伴奏',guitar:'吉他',piano:'钢琴'})[v.voice]||v.voice])]);$('voice').value='mix';$('voice').disabled=!s.analysis?.separation;
  lastClockShift=v?.audio_shift_ms??s.audio_shift_ms??(s.analysis?0:22.5);$('audioShift').value=lastClockShift;
  const tempo=v||s.analysis?.tempo;$('bpm').value=tempo?.bpm??120;$('offset').value=Math.round((tempo?.offset??0)-(v||!s.analysis?0:clockShift()));
  clockAnchor=!v&&s.analysis?s.analysis.tempo.offset:Number($('offset').value)+clockShift();
  if(changed)$('auditionOffset').value=$('offset').value;$('timingConfirmed').checked=!!v?.timing_confirmed;$('division').value=tempo?.division||4;
  tempoChoices=[...(s.analysis?.tempo?.candidates||[]),...(s.analysis?.tempo?.precise_candidates||[])].map(c=>({...c,audio_offset:c.offset,offset:Math.round(c.offset-clockShift())}));showTempoChoices();
  $('tempoNote').textContent=s.analysis?.tempo?.evidence_source==='drums'&&!s.analysis.tempo.timing_coverage_revision?'旧版鼓声拍线未检查缺鼓段落，请重新分析后确认前奏和间奏。':s.analysis?.tempo?.reliability||'分析后校准 BPM / Offset，再生成谱面。';
  $('timingReport').textContent=s.analysis?`原始检测 Offset ${s.analysis.tempo.offset} ms · 制谱提前量 ${clockShift()} ms。起音融合 ${s.analysis.fused_attacks||0} 处 · 分析耗时 ${s.analysis.elapsed_seconds??'未知'} 秒 · ${s.analysis.cached?'复用音高缓存':'本次运行音高模型'}`:'';
  timingSections=v?.timing_sections||s.analysis?.tempo?.sections||[];
  renderTimingLocations();
  $('planText').textContent=s.plan?.explanation||'';
  const ref=s.reference_style;$('referenceStyleInfo').textContent=ref?`${ref.title} · ${ref.keys}K · LN ${(ref.ln_ratio*100).toFixed(1)}% · 中位长度 ${ref.ln_duration_ms['50']||0} ms · 峰值 ${ref.peak_nps} 物件/秒。${ref.warnings.join(' ')} ${ref.notice}`:'不复制原谱音符或时间，也不保证相同手感。';
  options($('variants'),s.variants.length?s.variants.map((v,i)=>[i,`候选 ${i+1} · ${v.settings.keys}K · seed ${v.settings.seed} · ${v.notes.length} 物件${v.region?' · 局部':''}${v.edits?.length?' · 已修正 '+v.edits.length+' 次':''}`]):[['','尚未生成']]);$('variants').value=s.active;
  [['exportOsu','osu'],['exportOsz','osz'],['exportJson','json']].forEach(([id,kind])=>{const a=$(id);a.classList.toggle('disabled',!v||(kind==='osz'&&!s.audio));a.title=kind==='osz'&&!s.audio?'请导入配套音频后导出谱包':'';if(v)a.href=`/api/projects/${s.id}/export/${kind}`;else a.removeAttribute('href');a.download='ManiaLab.'+kind;});
  const report=v?.report;const vals=report?[report.notes,`${(report.ln_ratio*100).toFixed(1)}%`,report.max_held,`${(report.longest_ln_ms/1000).toFixed(2)}s`]:['—','—','—','—'];document.querySelectorAll('.metrics strong').forEach((e,i)=>e.textContent=vals[i]);
  selected=null;$('inspector').textContent='点击音符查看起止时间与来源。';window.resetWaveDetail?.();setMode();updateGridStatus();renderReview();window.renderLNDesign?.();window.renderNoteEditor?.();window.renderRecommendations?.();window.renderStructure?.();window.renderProvidedStems?.();window.renderPianoDetector?.();draw();refreshRecent();
}
function updateGridStatus(){
  const v=variant(),audioMode=v?.mode==='audio';
  const mismatch=audioMode&&(['bpm','offset','division'].some(k=>Number($(k).value)!==v[k])||clockShift()!==(v.audio_shift_ms||0));
  const stale=audioMode&&(v.timing_revision||0)<3;
  const g=v?.report?.grid;
  const diagnosis=$('exportJson'),hasDiagnosis=!!(v||state?.analysis);
  diagnosis.classList.toggle('disabled',!hasDiagnosis);if(hasDiagnosis)diagnosis.href=`/api/projects/${state.id}/export/json`;else diagnosis.removeAttribute('href');
  $('gridReport').textContent=stale?'旧版候选：请重新分析并生成整谱，才能导出。':mismatch?'当前 BPM / Offset / 细分与预览候选不同，请重新生成后导出。':g?`落线检查：${g.endpoints-g.off_grid} / ${g.endpoints} 个短键起点与 LN 起止在网格上；最大舍入误差 ${g.max_error_ms} ms。落线检查不判断采音是否正确。`:'';
  ['exportOsu','exportOsz'].forEach(id=>{const a=$(id),blocked=!v||v.report?.valid===false||stale||mismatch||(id==='exportOsz'&&!state?.audio);a.classList.toggle('disabled',blocked);if(blocked)a.removeAttribute('href');else a.href=`/api/projects/${state.id}/export/${id==='exportOsu'?'osu':'osz'}`;});
}
async function refreshRecent(){const status=await api('/api/status');window.updatePianoStatus?.(status.piano);separatorAvailable=!!status.separator?.available;$('analysisEngine').querySelector('[value=deep]').disabled=!separatorAvailable;$('analysisEngine').querySelector('[value=deep6]').disabled=!status.separator?.profiles?.six?.available;$('separationStatus').textContent=separatorAvailable?'本机 CPU 分离，再逐声部采音；六声部增加吉他与钢琴，效果请与四声部对照':'当前包未包含四声部组件 · 可使用快速采音';if(!state)$('analysisEngine').value=separatorAvailable?'deep':'basic';options($('recent'),[['','打开已保存项目…'],...status.projects.map(p=>[p.id,p.title])]);if(state)$('recent').value=state.id;$('engineStatus').textContent=status.engine.installed?`运行环境已就绪 · ${status.engine.model_downloaded?'模型权重已下载':'模型权重将在首次生成时下载'}`:'原模型未安装 · 不影响基础采音与定轨';$('priorStatus').textContent=status.pattern_prior?.available?`通过验证的键数：${status.pattern_prior.keys.map(k=>k+'K').join(' / ')}。仅提供定轨软偏好；未通过的键数自动忽略。`:'当前未加载经过验证的键型先验。';}
function setMode(){const remap=$('mode').value==='remap';$('musical').classList.toggle('hidden',remap);$('strictHint').classList.toggle('hidden',!remap);$('bpm').disabled=remap||busy;$('offset').disabled=remap||busy;['tempoCandidates','halfBpm','doubleBpm','division','timingConfirmed','checkTiming','refitTiming','referenceTiming','applyTiming','audioShift','waveAlign','auditionApply'].forEach(id=>$(id).disabled=remap||busy);$('generate').firstChild.textContent=remap?'严格重新定轨 ':'生成候选谱面 ';}
function setBusy(value){busy=value;['analyze','generate','another','regenerate','plan','charts','variants','recent','files','clearPlan','engineInstall','engineGenerate','referenceStyleFile','mode','preset','lnPreset','focus','layoutEngine','analysisEngine',...numbers].forEach(id=>$(id).disabled=value);document.querySelectorAll('[data-key]').forEach(button=>button.disabled=value);$('job').classList.toggle('hidden',!value);setMode();window.updateEditorLocks?.();window.lockStructure?.();window.lockProvidedStems?.();window.lockPianoDetector?.();document.querySelectorAll('[data-recommendation],#recommendationRefresh').forEach(b=>b.disabled=value||!state?.analysis);}
async function runJob(action,payload,form=false){
  if(!state)throw Error('请先导入素材');if(busy)return;setBusy(true);
  try{const job=await api(`/api/projects/${state.id}/${action}`,payload,form);jobId=job.id;
    while(true){const j=await api('/api/jobs/'+jobId);$('progress').value=j.progress;$('jobText').textContent=j.message;
      if(['done','error','cancelled'].includes(j.status)){if(j.status==='error')throw Error(j.message);if(j.status==='done'){loadState(await api('/api/projects/'+state.id));const v=variant();msg(action==='provided-stems'?'独立声轨已保存。点击分析音频，逐乐器检查采音与音乐分段。':action==='analyze'?`已提取 ${state.analysis.candidates.length} 个采音候选。切换「采音证据」试听检查。`:action==='structure'?`已划分 ${state.analysis.structure.segments.length} 个音乐段落，请检查声部表达方案。`:action==='plan'?'编排建议已应用，点击生成即可试谱。':action==='engine-install'?'原模型运行环境已安装。首次生成将自动下载 mania 权重。':`${v.report.notes} 个物件，${v.report.ln} 个 LN；${v.report.valid?'结构检查通过':`发现 ${v.report.issue_count} 项结构冲突，请先检查`}。${v.selection.message||''}\n编排耗时 ${j.elapsed_seconds??v.elapsed_seconds??'未知'} 秒。${v.selection.feasible_ln_ratio!==undefined?` 可达 LN 比例${v.selection.maximum_proven?'上限':'（已求得）'} ${(v.selection.feasible_ln_ratio*100).toFixed(1)}%。`:''}`);}else msg(j.message);return j.status;}
      await new Promise(r=>setTimeout(r,550));
    }
  }finally{jobId=null;setBusy(false);}
}
async function importFiles(files){if(!files.length||busy)return;setBusy(true);$('jobText').textContent='导入素材…';try{const data=new FormData();for(const f of files)data.append('files',f);loadState(await api('/api/import',data,true));msg(state.mode==='remap'?'原谱已导入，可直接严格重新定轨。':'音频已导入，点击分析音频开始。');}finally{setBusy(false);$('files').value='';}}
async function generate(region){$('offset').value=Math.round(Number($('offset').value));await runJob('generate',{settings:settings(),mode:$('mode').value,bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value),audio_shift_ms:clockShift(),snap:true,timing_confirmed:$('timingConfirmed').checked,...(region?{region}:{})});}
numbers.forEach(k=>$(k).addEventListener('input',updateOutputs));
document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{keys=Number(b.dataset.key);updateKeys();});
$('referenceStyleFile').onchange=guarded(async e=>{
  const file=e.target.files[0];if(!file)return;if(!state)throw Error('请先导入要制谱的音频或项目');if(busy)return;
  const fields=['bpm','offset','division','audioShift','auditionOffset','mode','analysisEngine'];
  const draft=Object.fromEntries(fields.map(id=>[id,$(id).value])),confirmed=$('timingConfirmed').checked,anchor=clockAnchor;
  setBusy(true);
  try{
    const data=new FormData();data.append('file',file);data.append('settings',JSON.stringify(settings()));
    loadState(await api(`/api/projects/${state.id}/reference-style`,data,true));
    fields.forEach(id=>$(id).value=draft[id]);lastClockShift=clockShift();clockAnchor=anchor;$('timingConfirmed').checked=confirmed;
    tempoChoices=tempoChoices.map(c=>({...c,offset:Math.round(c.audio_offset-clockShift())}));showTempoChoices();
    setMode();updateGridStatus();draw();
    msg('已套用参考谱的全局偏好，请重新生成候选。已有 API 分段建议仍会应用，可按需清除。');
  }finally{setBusy(false);$('referenceStyleFile').value='';}
});
$('files').onchange=guarded(e=>importFiles(e.target.files));
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('drag');};$('drop').ondragleave=()=>$('drop').classList.remove('drag');$('drop').ondrop=guarded(e=>{e.preventDefault();$('drop').classList.remove('drag');return importFiles(e.dataTransfer.files);});
$('recent').onchange=guarded(async e=>{if(e.target.value)loadState(await api('/api/projects/'+e.target.value));});
$('charts').onchange=guarded(async e=>{audio.removeAttribute('src');loadState(await api(`/api/projects/${state.id}/chart`,{path:e.target.value}));});
$('variants').onchange=guarded(async e=>loadState(await api(`/api/projects/${state.id}/variant`,{index:Number(e.target.value)})));
$('analyze').onclick=guarded(()=>runJob('analyze',{settings:settings(),engine:$('analysisEngine').value,piano_engine:window.pianoSelection?.()||'general',audio_shift_ms:clockShift()}));$('generate').onclick=guarded(()=>generate());
$('another').onclick=guarded(()=>{$('seed').value=Math.floor(Math.random()*2147483647);return generate();});
$('regenerate').onclick=guarded(()=>generate([Number($('regionStart').value)*1000,Number($('regionEnd').value)*1000]));
$('cancel').onclick=guarded(()=>jobId?api('/api/jobs/'+jobId+'/cancel',{}):null);
$('mode').onchange=setMode;
$('plan').onclick=guarded(()=>runJob('plan',{endpoint:$('endpoint').value,model:$('model').value,key:$('apiKey').value,instruction:$('instruction').value,settings:settings()}));
$('clearPlan').onclick=guarded(async()=>{if(state)loadState(await api(`/api/projects/${state.id}/clear-plan`,{}));});
$('engineInstall').onclick=guarded(()=>runJob('engine-install',{}));
$('engineGenerate').onclick=guarded(()=>runJob('engine-generate',{settings:settings(),seconds:Number($('engineSeconds').value),use_timing:$('engineUseTiming').checked,bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value),audio_shift_ms:clockShift(),timing_confirmed:$('timingConfirmed').checked}));
$('preset').onchange=()=>{const presets={balanced:{density:7,ln_ratio:.55,min_ln:240,max_held:5,chord_size:4,focus:'balanced'},sustain:{density:6,ln_ratio:.85,min_ln:300,max_held:4,chord_size:3,focus:'pitched'},rhythm:{density:8,ln_ratio:.25,min_ln:240,max_held:3,chord_size:4,focus:'transient'}};setSettings({...settings(),...presets[$('preset').value]});};
function setView(v){cueClock.reset();view=v;selected=null;window.renderNoteEditor?.();$('chartView').classList.toggle('active',v==='chart');$('evidenceView').classList.toggle('active',v==='evidence');draw();}
$('chartView').onclick=()=>setView('chart');$('evidenceView').onclick=()=>setView('evidence');$('window').onchange=draw;
$('helpBtn').onclick=()=>$('help').showModal();$('closeHelp').onclick=()=>$('help').close();
function seek(t){manualTime=Math.max(0,Math.min(duration(),t));if(audio.src&&Number.isFinite(audio.duration))audio.currentTime=manualTime;lastAudioTime=manualTime;draw();}
$('seek').oninput=()=>seek(Number($('seek').value));$('waveform').onclick=e=>{const r=$('waveform').getBoundingClientRect();seek((e.clientX-r.left)/r.width*duration());};
function canvas(id){const el=$(id),r=el.getBoundingClientRect(),d=window.devicePixelRatio||1;if(el.width!==Math.round(r.width*d)||el.height!==Math.round(r.height*d)){el.width=Math.round(r.width*d);el.height=Math.round(r.height*d);}const c=el.getContext('2d');c.setTransform(d,0,0,d,0,0);return[c,r.width,r.height];}
function fmt(t){return `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(3).padStart(6,'0')}`;}
function draw(){
  const v=variant();displayed=(view==='evidence'?(state?.analysis?.candidates||[]):(v?.notes||state?.input_notes||[])).filter(n=>$('voice').value==='mix'||n.voice===$('voice').value);const t=currentTime(),span=Number($('window').value);const total=duration();$('seek').max=Math.max(1,total);$('seek').value=t;$('clock').textContent=fmt(t);$('emptyState').classList.toggle('hidden',displayed.length>0);
  const [w,ww,wh]=canvas('waveform');w.clearRect(0,0,ww,wh);w.fillStyle='#344e61';const wave=state?.analysis?.waveform||[];wave.forEach((a,i)=>{const x=i/wave.length*ww;w.fillRect(x,wh/2-a*wh*.44,Math.max(1,ww/wave.length),Math.max(1,a*wh*.88));});for(const section of timingSections){if(!section.suspect||!total)continue;w.fillStyle='#dc795f';w.fillRect(section.start_ms/1000/total*ww,wh-4,(section.end_ms-section.start_ms)/1000/total*ww,4);}w.fillStyle='#78e4c4';w.fillRect(total?t/total*ww:0,0,2,wh);
  const [c,width,height]=canvas('roll');c.clearRect(0,0,width,height);const k=view==='evidence'?88:(v?.settings.keys||state?.input_keys||keys);const left=view==='evidence'?28:36;const cw=(width-left-12)/k;const bottom=height-40;const scale=(height-75)/span;hitboxes=[];
  if(view==='evidence'){c.fillStyle='#627d94';c.font='10px Segoe UI';c.fillText('低音 ← 音高 → 高音',left+5,17);}
  for(let lane=0;lane<k;lane++){c.fillStyle=lane%2===0?'#142131':'#101b29';c.fillRect(left+lane*cw,25,cw-1,height-25);if(view==='chart'){c.fillStyle='#8199af';c.font='11px Segoe UI';c.textAlign='center';c.fillText(String(lane+1),left+(lane+.5)*cw,height-12);}}
  c.textAlign='left';c.font='10px Segoe UI';for(let second=Math.ceil(t);second<t+span;second++){const y=bottom-(second-t)*scale;c.strokeStyle='#27374b';c.beginPath();c.moveTo(left,y);c.lineTo(width-12,y);c.stroke();c.fillStyle='#718399';c.fillText(`${second}s`,3,y+3);}
  drawBeatGrid(c,left,width,bottom,scale,t,span);
  for(const n of displayed){if(n.end/1000<t-.2||n.start/1000>t+span)continue;const lane=view==='evidence'?(n.source==='transient'?2:n.pitch-21):n.lane;if(lane<0||lane>=k)continue;const x=left+lane*cw+Math.min(3,cw*.1),bw=Math.max(2,cw-Math.min(6,cw*.2));const y=bottom-(n.start/1000-t)*scale,ye=bottom-(n.end/1000-t)*scale;const hold=n.end>n.start;c.fillStyle=n.id===selected?'#ffffff':hold?'#78e4c4':'#ffb978';c.globalAlpha=hold?.8:.95;const top=Math.max(25,ye),end=Math.min(bottom+5,y);if(end>=25&&top<=bottom+5){c.fillRect(x,hold?top:end-4,bw,hold?Math.max(5,end-top):5);hitboxes.push({n,x,y:hold?top:end-6,w:bw,h:hold?Math.max(7,end-top):9});}c.globalAlpha=1;}
  c.strokeStyle='#d4f9ef';c.lineWidth=1.5;c.beginPath();c.moveTo(left,bottom);c.lineTo(width-12,bottom);c.stroke();
}
function inspectNote(n){
  selected=n.id;window.renderNoteEditor?.(n);
  const movement=variant()?.edits?.some(e=>e.after?.id===n.id)?'修正后与检测差':'量化移动';
  const shift=n.evidence_start===null||n.evidence_start===undefined?'':` · 制谱基准下的原始起音 ${n.evidence_start} ms（${movement} ${n.start-n.evidence_start>0?'+':''}${n.start-n.evidence_start} ms）`;
  const tail=n.end>n.start&&Number.isFinite(n.evidence_end)?` · 检测音尾 ${n.evidence_end} ms（${movement} ${n.end-n.evidence_end>0?'+':''}${n.end-n.evidence_end} ms）`:'';
  $('inspector').textContent=`${n.end>n.start?'LN / 持续音':'短键 / 瞬态'} · 起点 ${n.start} ms · 终点 ${n.end} ms · 长度 ${n.end-n.start} ms · ${n.source==='import'?'原谱物件':n.source==='pitch'?`检测音高 MIDI ${n.pitch} · 激活强度 ${n.strength}`:'频谱瞬态'}${n.lane>=0?' · 轨道 '+(n.lane+1):''} · 声部 ${n.voice||'mix'}${n.phrase?' · 跨音高持续段 · 支持 '+((n.support_coverage||0)*100).toFixed(0)+'%':''}${shift}`;
  $('inspector').textContent+=tail+(n.phrase?` · 支撑音高候选 ${(n.support||[]).length} 个`:'');
  if(n.release_reason){const reasons={release_regression:'检测到释放',frame_decay:'持续激活消退',reattack:'同音高重新起音',duration_limit:'达到模型时长上限，未确认音尾',input_boundary:'到达音频边界，未确认音尾'};$('inspector').textContent+=' · 音尾依据：'+(reasons[n.release_reason]||n.release_reason);}
  draw();
}
$('roll').onclick=e=>{const rect=$('roll').getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=[...hitboxes].reverse().find(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);if(hit)inspectNote(hit.n);};
const cueClock=new ManiaAudioClock(audio,(from,to)=>{
  const events=[];
  if($('metronome').checked)events.push(...beatsBetween(from,to).map(b=>({time:b.time,kind:b.accent?'accent':'beat'})));
  const notes=(view==='evidence'?(state?.analysis?.candidates||[]):(variant()?.notes||state?.input_notes||[])).filter(n=>$('voice').value==='mix'||n.voice===$('voice').value);
  if($('clicks').checked)for(const n of notes)if(n.start/1000>=from&&n.start/1000<to)events.push({time:n.start/1000,kind:'head'});
  if($('releases').checked)for(const n of notes)if(n.end>n.start&&n.end/1000>=from&&n.end/1000<to)events.push({time:n.end/1000,kind:'tail'});
  return events;
});
function tick(){if(!audio.paused)draw();requestAnimationFrame(tick);}
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
function timingChanged(preserveClockAnchor=false){ if(!preserveClockAnchor)clockAnchor=Number($('offset').value)+clockShift();cueClock.reset();$('timingConfirmed').checked=false;timingSections=[];renderTimingLocations();lastScheduledBeat=-Infinity;updateGridStatus();draw();window.drawWaveDetail?.(); }
['bpm','offset','division'].forEach(id=>$(id).addEventListener('input',()=>timingChanged()));
$('audioShift').addEventListener('change',()=>{const value=clockShift();if(!Number.isFinite(value)||Math.abs(value)>100)return;$('offset').value=Math.round(clockAnchor-value);lastClockShift=value;tempoChoices=tempoChoices.map(c=>({...c,offset:Math.round(c.audio_offset-value)}));showTempoChoices();timingChanged(true);msg(`制谱提前量 ${value} ms；Offset 已同步调整。原始采音保留，需重新生成整谱。`);});
$('tempoCandidates').onchange=()=>{const i=$('tempoCandidates').value;if(i==='')return;const c=tempoChoices[Number(i)];$('bpm').value=c.bpm;$('offset').value=c.offset;$('division').value=c.division;timingChanged();if(c.precision_suggestion)msg(`已采用 ${c.bpm} BPM 精确候选；请在歌曲开头和结尾试听，确认没有逐渐漂移。`);};
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
  const pid=state.id,bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value),shift=clockShift();
  const r=await api(`/api/projects/${pid}/timing-refit`,{bpm,offset,division,audio_shift_ms:shift});
  if(state.id!==pid||Number($('bpm').value)!==bpm||Number($('offset').value)!==offset||Number($('division').value)!==division||clockShift()!==shift)return;
  $('offset').value=r.offset;timingChanged();timingSections=r.sections;renderTimingLocations();draw();
  $('timingReport').textContent=`按 ${r.bpm} BPM 重新估计 Offset：${r.offset} ms。${r.suspect} 段可能错位；请试听确认。`;
});
$('checkTiming').onclick=guarded(async()=>{
  if(!state)throw Error('请先分析音频');
  const pid=state.id,bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value),shift=clockShift();
  const r=await api(`/api/projects/${pid}/timing-check`,{bpm,offset,division,audio_shift_ms:shift});
  if(state.id!==pid||Number($('bpm').value)!==bpm||Number($('offset').value)!==offset||Number($('division').value)!==division||clockShift()!==shift)return;
  timingSections=r.sections;renderTimingLocations();const suspicious=r.sections.filter(s=>s.suspect);$('timingReport').textContent=`检查 ${r.sections.length} 段，${r.suspect} 段可能错位。`+(suspicious.length?'请定位试听：'+suspicious.slice(0,8).map(s=>`${(s.start_ms/1000).toFixed(0)}–${(s.end_ms/1000).toFixed(0)}s（相位偏差 ${s.phase_error_ms}ms）`).join('；'):'请再用节拍器确认倍速、半速和弱起。');draw();
});
['metronome','clicks','releases'].forEach(id=>$(id).onchange=guarded(async()=>{cueClock.reset();if($(id).checked)await cueClock.enable();}));
let auditionUrl=null,auditionRequest=0;
$('voice').onchange=()=>{if(!state)return;const t=currentTime();cueClock.reset();audio.pause();audio.src=$('voice').value==='mix'?`/api/projects/${state.id}/audio`:`/api/projects/${state.id}/stem/${$('voice').value}`;audio.addEventListener('loadedmetadata',()=>{audio.currentTime=Math.min(t,audio.duration||t);draw();},{once:true});draw();};
$('auditionHere').onclick=()=>{$('auditionStart').value=currentTime().toFixed(1);};
$('auditionLoop').onchange=()=>{$('auditionAudio').loop=$('auditionLoop').checked;};
$('auditionApply').onclick=()=>{$('offset').value=Math.round(Number($('auditionOffset').value));timingChanged();msg('已采用对照 B 的 Offset；请检查整曲后重新生成。');};
async function audition(mode,compare=false){
  if(!state?.analysis)throw Error('请先分析音频');
  const requestNumber=++auditionRequest;
  const pid=state.id,offset=Number($(compare?'auditionOffset':'offset').value);
  const payload={mode,start_ms:Number($('auditionStart').value)*1000,duration_ms:Number($('auditionLength').value)*1000,
    bpm:Number($('bpm').value),offset,division:Number($('division').value),audio_shift_ms:clockShift(),releases:true,voice:$('voice').value};
  $('auditionNote').textContent='正在混合音乐和提示音…';
  const response=await fetch(`/api/projects/${pid}/audition`,{method:'POST',headers:{'X-ManiaLab-Token':token,'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!response.ok)throw Error((await response.json()).error||'试听失败');
  const blob=await response.blob();if(state.id!==pid||requestNumber!==auditionRequest)return;
  audio.pause();$('auditionAudio').pause();if(auditionUrl)URL.revokeObjectURL(auditionUrl);
  auditionUrl=URL.createObjectURL(blob);$('auditionAudio').src=auditionUrl;$('auditionAudio').loop=$('auditionLoop').checked;
  $('auditionNote').textContent=mode==='beats'?`${compare?'B':'A'}：${payload.bpm} BPM / ${offset} ms，${payload.start_ms/1000} 秒开始。音乐和节拍器共用采样时钟。`:
    `${mode==='chart'?'候选谱面':'原始采音'}：${payload.start_ms/1000} 秒开始，高音提示起点、低音提示持续音结束。`;
  await $('auditionAudio').play();
}
$('auditionA').onclick=guarded(()=>audition('beats'));$('auditionB').onclick=guarded(()=>audition('beats',true));
$('auditionChart').onclick=guarded(()=>audition('chart'));$('auditionEvidence').onclick=guarded(()=>audition('evidence'));
audio.addEventListener('play',()=>{$('auditionAudio').pause();});
updateOutputs();guarded(refreshRecent)();draw();requestAnimationFrame(tick);
