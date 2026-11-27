'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="manialab-token"]').content;
const numbers = ['density','ln_ratio','min_ln','max_held','release_gap','chord_size','pitch_follow','balance','jack_avoid','seed','onset_threshold','frame_threshold','phrase_ln','pattern_prior','drums_weight','bass_weight','vocals_weight','other_weight','guitar_weight','piano_weight','ln_coordination','ln_release','ln_inverse'];
let timingSections=[], tempoChoices=[], lastScheduledBeat=-Infinity,separatorAvailable=false;
let mapperInstallAvailable=false,mapperGenerateAvailable=false;
let v32GenerateAvailable=false,v32EngineStatus=null;
let singleEngineTouched=false,batchEngineTouched=false;
let timingDiagnostics=null,timingRequest=0;
let lastClockShift=22.5;
let clockAnchor=22.5;
const clockShift=()=>Number($('audioShift').value);
const timingLocations=document.createElement('div');timingLocations.id='timingLocations';$('timingReport').after(timingLocations);
let state = null, keys = 10, view = 'chart', busy = false, jobId = null, displayed = [], hitboxes = [], selected = null, manualTime = 0, audioContext = null, lastAudioTime = 0, projectLoadRequest = 0, auditionUrl = null, auditionRequest = 0, playbackSourceRequest = 0, playbackMetadataHandler = null;
let batchJobId = null, batchImported = [], batchSnapshot = null;
let batchAnalysisSnapshot = null;
let batchImporting = false, batchGenerationSubmitting = false, batchCancelRequested = false, batchPhase = 'analysis';
let readyBatchProjectIds = new Set(), previewController=null, localLoras=[];
let batchGenerationAttemptIds=new Set(),batchAttemptJobId=null;
let batchResultsOnly=false;
const PRESET_V32_HOLD={rice_stream:.12,rice_jack:.12,hybrid_adaptive:.3,ln_focus:.55};
const PRESET_TOROSU_DEFAULTS={rice_stream:'dense',rice_jack:'more-dense',hybrid_adaptive:'dense',ln_focus:'dense',torosu_remap:'none'};
const KEY_LAYOUT_CHOICES={
  6:[['6k_3_3','6K · 3+3']],
  7:[['7k_3_1_3','7K · 3+1+3']],
  8:[['8k_3_2_3','8K · 3+2+3'],['8k_4_4','8K · 4+4']],
  9:[['9k_4_1_4','9K · 4+1+4']],
  10:[['10k_5_5','10K · 5+5']],
};
const DEFAULT_KEY_LAYOUT={6:'6k_3_3',7:'7k_3_1_3',8:'8k_3_2_3',9:'9k_4_1_4',10:'10k_5_5'};
const GENERATION_PRESETS={
  rice_stream:{density:8,ln_ratio:.12,phrase_ln:.2,min_ln:180,max_held:3,chord_size:3,focus:'transient',layout_engine:'torosu',torosu_keep_jacks:false,torosu_density:'none',torosu_ln_mode:'keep'},
  rice_jack:{density:8,ln_ratio:.12,phrase_ln:.2,min_ln:180,max_held:4,chord_size:5,focus:'balanced',layout_engine:'torosu',torosu_keep_jacks:true,torosu_density:'none',torosu_ln_mode:'keep'},
  hybrid_adaptive:{density:7,ln_ratio:.3,phrase_ln:.45,min_ln:240,max_held:5,chord_size:4,focus:'balanced',layout_engine:'torosu',torosu_keep_jacks:true,torosu_density:'none',torosu_ln_mode:'keep'},
  ln_focus:{density:6,ln_ratio:.55,phrase_ln:.75,min_ln:300,max_held:4,chord_size:3,focus:'pitched',layout_engine:'torosu',torosu_keep_jacks:true,torosu_density:'none',torosu_ln_mode:'keep'},
  torosu_remap:{density:7,ln_ratio:.55,phrase_ln:.65,min_ln:240,max_held:5,chord_size:4,focus:'balanced',layout_engine:'torosu',torosu_keep_jacks:true,torosu_density:'none',torosu_ln_mode:'keep'},
  balanced:{density:7,ln_ratio:.55,phrase_ln:.65,min_ln:240,max_held:5,chord_size:4,focus:'balanced'},
  onset_only:{density:8,ln_ratio:0,phrase_ln:0,min_ln:240,max_held:5,chord_size:4,focus:'balanced'},
  sustain:{density:6,ln_ratio:.85,phrase_ln:.85,min_ln:300,max_held:4,chord_size:3,focus:'pitched'},
  rhythm:{density:8,ln_ratio:.25,phrase_ln:.25,min_ln:240,max_held:3,chord_size:4,focus:'transient'},
};
const PRESET_SETTING_KEYS=['density','ln_ratio','phrase_ln','min_ln','max_held','chord_size','focus','layout_engine','torosu_keep_jacks','torosu_density','torosu_ln_mode'];
const CANDIDATE_SETTING_KEYS=['keys','key_layout','focus','layout_engine','torosu_keep_jacks','torosu_density','torosu_ln_mode',
  ...numbers.filter(key=>!['onset_threshold','frame_threshold'].includes(key))];
const audio = $('audio');
const msg = (text, error=false) => { $('message').textContent=text; $('message').classList.toggle('error',error); };
async function api(path, data, form=false) {
  const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'X-ManiaLab-Token':token,...(form?{}:{'Content-Type':'application/json'})},body:data===undefined?undefined:form?data:JSON.stringify(data)});
  const result=await response.json().catch(()=>({})); if(!response.ok){const error=new Error(result.error||'请求失败');error.status=response.status;throw error;} return result;
}
const guarded=fn=>async(...args)=>{try{await fn(...args)}catch(e){msg(e.message,true)}};
function settings(){const p={keys,key_layout:$('keyLayout').value,focus:$('focus').value,layout_engine:$('layoutEngine').value,torosu_keep_jacks:$('torosuKeepJacks').checked,torosu_density:$('torosuLegacyDensity').value,torosu_ln_mode:$('torosuLnMode').value};numbers.forEach(k=>p[k]=Number($(k).value));return p;}
function updateOutputs(){numbers.forEach(k=>{if($(k+'Out'))$(k+'Out').textContent=['ln_ratio','phrase_ln','pattern_prior','ln_coordination','ln_release','ln_inverse'].includes(k)?`${Math.round(Number($(k).value)*100)}%`:$(k).value;});if($('torosuKeepJacks'))$('torosuKeepJacks').checked=settings().torosu_keep_jacks;}
function matchingPreset(p){
  return Object.entries(GENERATION_PRESETS).filter(([id])=>['rice_stream','rice_jack','hybrid_adaptive','ln_focus','torosu_remap'].includes(id)).find(([,preset])=>
    PRESET_SETTING_KEYS.every(key=>p[key]===preset[key]))?.[0]||'custom';
}
function syncPresetLabel(p=settings()){
  const preset=$('preset');if(preset)preset.value=matchingPreset(p);
}
function candidateSettingsMismatch(candidate){
  if(!candidate?.settings)return false;
  const draft=settings();
  return CANDIDATE_SETTING_KEYS.some(key=>candidate.settings[key]!==draft[key]);
}
function setSettings(p){keys=p.keys;numbers.forEach(k=>{$(k).value=p[k]??$(k).defaultValue;});$('layoutEngine').value=p.layout_engine||'beam';$('focus').value=p.focus||'balanced';$('torosuKeepJacks').checked=p.torosu_keep_jacks!==false;$('torosuLegacyDensity').value=p.torosu_density||'none';$('torosuLnMode').value=p.torosu_ln_mode||'keep';updateOutputs();updateKeys(p.key_layout);syncPresetLabel(p);}
function updateKeyLayoutOptions(requested){const choices=KEY_LAYOUT_CHOICES[keys]||[];const previous=requested||$('keyLayout').value;options($('keyLayout'),choices);$('keyLayout').value=choices.some(([value])=>value===previous)?previous:(DEFAULT_KEY_LAYOUT[keys]||choices[0]?.[0]||'');}
function updateKeys(keyLayout){document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',Number(b.dataset.key)===keys));$('keytag').replaceChildren(document.createTextNode(keys));const s=document.createElement('span');s.textContent='KEYS';$('keytag').append(s);updateKeyLayoutOptions(keyLayout);draw();}
function options(el,items){el.replaceChildren(...items.map(([value,text])=>{const o=document.createElement('option');o.value=value;o.textContent=text;return o;}));}
function showTempoChoices(){options($('tempoCandidates'),[['','选择其他节拍解释…'],...tempoChoices.map((c,i)=>[i,`${c.precision_suggestion?'精确候选 · ':''}${c.bpm} BPM · 偏移 ${c.offset} ms · 匹配 ${(c.coverage*100).toFixed(0)}%${c.precision_suggestion?` · 取整漂移约 ${c.rounding_drift_ms}ms`:''}`])]);}
function variant(){return state&&state.active>=0?state.variants[state.active]:null;}
function candidateExportUrl(kind,v){
  return `/api/projects/${state.id}/export/${kind}?candidate_id=${encodeURIComponent(v?.candidate_id||'')}`;
}
function duration(){return Math.max(state?.analysis?.duration||0,Number.isFinite(audio.duration)?audio.duration:0,displayed.reduce((m,n)=>Math.max(m,n.end/1000),0),(state?.input_notes||[]).reduce((m,n)=>Math.max(m,n.end/1000),0));}
function currentTime(){return audio.src&&audio.readyState>=audio.HAVE_METADATA&&Number.isFinite(audio.currentTime)?audio.currentTime:manualTime;}
function cancelAuditionPlayback(){
  auditionRequest++;$('auditionAudio').pause();$('auditionAudio').removeAttribute('src');
  if(auditionUrl){URL.revokeObjectURL(auditionUrl);auditionUrl=null;}
}
function playbackSeparation(){
  const candidate=variant();
  if(view==='chart'&&candidate?.mode==='audio')return candidate.source_separation?.separation||null;
  return state?.analysis?.separation||null;
}
function playbackSourceIdentity(){
  const candidate=variant();
  if(view==='chart'&&candidate?.mode==='audio')
    return candidate.source_separation?.identity||'';
  return playbackSeparation()?.playback_identity||'';
}
function playbackVoices(separation,identity){
  if(!separation||!identity)return [];
  const voices=separation.source==='provided'?(separation.voices||[]):
    (separation.profile==='six'?['drums','bass','vocals','other','guitar','piano']:
      separation.profile==='four-piano'?['drums','piano','vocals','other']:['drums','bass','vocals','other']);
  return voices.filter(voice=>separation.source==='provided'
    ?!!separation.decoded_files?.[voice]?.wav_sha256
    :separation.review_audio_identity===identity&&!!separation.review_audio_files?.[voice]);
}
function updateVoiceOptions(reset=false){
  const separation=playbackSeparation();
  const voices=playbackVoices(separation,playbackSourceIdentity());
  const selected=reset?'mix':$('voice').value;
  const names={drums:'鼓 / 打击',bass:'贝斯',vocals:'人声',other:'其余伴奏',guitar:'吉他',piano:'钢琴'};
  options($('voice'),[['mix','全部声部'],...voices.map(voice=>[voice,names[voice]||voice])]);
  $('voice').value=voices.includes(selected)?selected:'mix';$('voice').disabled=!voices.length;
}
function playbackSourceUrl(){
  const voice=$('voice').value||'mix';
  if(voice!=='mix')return `/api/projects/${state.id}/stem/${voice}?context=${view}&source_identity=${encodeURIComponent(playbackSourceIdentity())}`;
  return `/api/projects/${state.id}/audio?audio=${encodeURIComponent(state.audio||'')}&revision=${state.analysis?.revision||0}&elapsed=${state.analysis?.elapsed_seconds||0}`;
}
function updateAudioStatus(){
  if(!state)return;
  const warnings=[];
  if(state.mode==='remap'&&state.original&&!state.audio)
    warnings.push('此难度引用的音频未导入。请重新导入此 .osu 文件和对应音频；不会自动借用其他难度的音轨。');
  if(view==='chart'&&variant()?.mode==='audio'&&!variant()?.source_separation)
    warnings.push('此旧候选未保存声轨来源；声部试听和波形核证已停用。切到「采音证据」可试听当前分析，或重新生成候选。');
  $('audioStatus').textContent=warnings.join(' ');
  $('audioStatus').classList.toggle('hidden',!warnings.length);
}
function setPlaybackSource(resetTime=false){
  const src=state?.audio?playbackSourceUrl():null;
  if(src&&audio.getAttribute('src')===src)return;
  const wasPlaying=!audio.paused;
  manualTime=resetTime?0:currentTime();
  const request=++playbackSourceRequest;
  if(playbackMetadataHandler)audio.removeEventListener('loadedmetadata',playbackMetadataHandler);
  playbackMetadataHandler=null;
  audio.pause();
  if(!src){audio.removeAttribute('src');audio.load();return;}
  playbackMetadataHandler=()=>{
    if(request!==playbackSourceRequest||audio.getAttribute('src')!==src)return;
    playbackMetadataHandler=null;
    audio.currentTime=Math.min(manualTime,audio.duration||manualTime);draw();
  };
  audio.addEventListener('loadedmetadata',playbackMetadataHandler,{once:true});
  audio.src=src;audio.load();
  // Let a later native pause cancel playback while this source is loading.
  if(wasPlaying)audio.play().catch(error=>{
    if(request===playbackSourceRequest&&error.name!=='AbortError')msg('音频播放失败：'+error.message,true);
  });
}
function currentTimingClock(){return {bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value),audio_shift_ms:clockShift()};}
function sameTimingClock(clock){const current=currentTimingClock();return !!clock&&Object.keys(current).every(key=>clock[key]===current[key]);}
// Keep inferred clocks aligned with the server default; explicit edits use chartOffset.
function autoChartOffset(value){return Math.round(value);}
function captureTimingDraft(extraFields=[]){
  const fields=['bpm','offset','division','audioShift','auditionOffset','mode',...extraFields];
  return {values:Object.fromEntries(fields.map(id=>[id,$(id).value])),confirmed:$('timingConfirmed').checked,anchor:clockAnchor};
}
function restoreTimingDraft(draft){
  for(const [id,value] of Object.entries(draft.values))$(id).value=value;
  lastClockShift=clockShift();clockAnchor=draft.anchor;$('timingConfirmed').checked=draft.confirmed;
  // loadState may have displayed the saved candidate's diagnosis. Restoring a
  // different draft clock cannot reuse that result or its marked windows.
  if(timingDiagnostics&&!sameTimingClock(timingDiagnostics.clock)){clearTimingDiagnostics();renderTimingLocations();}
  tempoChoices=tempoChoices.map(c=>({...c,offset:autoChartOffset(c.audio_offset-clockShift())}));showTempoChoices();
  $('structurePanel')?.classList.toggle('hidden',$('mode').value==='remap');
  setMode();updateGridStatus();draw();
  // Invalidate the recommendation request made by loadState with its temporary
  // candidate BPM, and derive suggestions from the restored draft instead.
  window.renderRecommendations?.();
}
function chartOffset(value){
  const lower=Math.floor(value);
  return value-lower===.5?lower+(lower%2===0?0:1):Math.round(value);
}
function normalizeTimingOffset(){
  const value=Number($('offset').value);if(!Number.isFinite(value)||Math.abs(value)>600000)throw Error('Offset 必须在 -600000–600000 ms 内');
  const rounded=chartOffset(value);
  if(value!==rounded){$('offset').value=rounded;timingChanged();msg(`Offset 已取整为 ${rounded} ms；本次检查使用此拍线。`);}
}
function renderOriginalTempoNote(){
  const analysis=state?.analysis,tempo=analysis?.tempo;
  if(!tempo){$('tempoNote').textContent='分析后校准 BPM / Offset，再生成谱面。';return;}
  const explanation=tempo.evidence_source==='drums'&&!tempo.timing_coverage_revision?'旧版鼓声拍线未检查缺鼓段落，请重新分析后确认前奏和间奏。':tempo.reliability||'请试听核对原始估计。';
  $('tempoNote').textContent=`原始估计：${tempo.bpm} BPM · 原音频 Offset ${tempo.offset} ms · ${tempo.division||4} 等分。${explanation} 原始结论不代替当前拍线检查。起音融合 ${analysis.fused_attacks||0} 处 · 分析耗时 ${analysis.elapsed_seconds??'未知'} 秒 · ${analysis.cached?'复用音高缓存':'本次运行音高模型'}。`;
}
function clearTimingDiagnostics(){timingRequest++;timingDiagnostics=null;timingSections=[];$('timingReport').textContent=state?.analysis?'当前拍线尚未检查，请检查整曲并试听确认。':'';}
function showTimingDiagnostics(diagnostics,prefix=''){
  if(diagnostics.clock&&!sameTimingClock(diagnostics.clock))return false;
  timingDiagnostics=diagnostics;timingSections=diagnostics.sections||[];
  const insufficient=diagnostics.insufficient_windows||[],suspect=timingSections.filter(s=>s.suspect);
  const unavailable=diagnostics.status==='insufficient'||!timingSections.length;
  const coverage=unavailable?'证据不足，不能判断当前拍线；没有可评估的窗口。':`检查 ${timingSections.length} 段，${suspect.length} 段可能错位。`;
  const missing=insufficient.length?`${insufficient.length} 个窗口证据不足，请单独试听这些区间。`:'';
  const detail=suspect.length?'请定位试听：'+suspect.slice(0,8).map(s=>`${(s.start_ms/1000).toFixed(0)}–${(s.end_ms/1000).toFixed(0)}s（相位偏差 ${s.phase_error_ms}ms）`).join('；'):'请试听确认倍速、半速和弱起。';
  $('timingReport').textContent=prefix+coverage+missing+detail;
  renderTimingLocations();draw();return true;
}
function renderTimingLocations(){
  timingLocations.replaceChildren();if(!state?.analysis)return;
  const length=state.analysis.duration||0,suspect=timingSections.filter(s=>s.suspect);
  const insufficient=timingDiagnostics?.insufficient_windows||[];
  const explanation=document.createElement('p');explanation.className='muted compact';explanation.textContent=suspect.length||insufficient.length?'点击可疑或证据不足区间，在对音实验室试听检查：':'建议分别试听开头、中间和结尾：';timingLocations.append(explanation);
  const locations=[...suspect.map(s=>({time:s.start_ms/1000,label:`${(s.start_ms/1000).toFixed(0)}–${Math.min(length,s.end_ms/1000).toFixed(0)}s`,title:`${s.fallback?'鼓点不足，使用混音核对 · ':''}相位偏差 ${s.phase_error_ms}ms · 起音匹配 ${(s.coverage*100).toFixed(0)}%`})),...insufficient.map(s=>({time:s.start_ms/1000,label:`证据不足 ${(s.start_ms/1000).toFixed(0)}–${Math.min(length,s.end_ms/1000).toFixed(0)}s`,title:`仅 ${s.attacks} 个起音，不能判断本窗口拍线`}))].slice(0,12);
  if(!locations.length)locations.push({time:0,label:'开头'},{time:Math.max(0,length/2-6),label:'中间'},{time:Math.max(0,length-12),label:'结尾'});
  for(const item of locations){
    const button=document.createElement('button');button.className='quiet';button.textContent=item.label;button.title=item.title||'定位到对音实验室';
    button.onclick=()=>{const time=Math.min(Math.max(0,length-1),item.time);audio.pause();seek(time);$('auditionStart').value=time.toFixed(1);const panel=document.querySelector('.audition-panel');panel.open=true;$('auditionNote').textContent=`已定位 ${time.toFixed(1)} 秒；点击「试听当前拍线 A」比较音乐与拍线，或设置 B 的 Offset 做对照。`;panel.scrollIntoView({block:'center',behavior:'smooth'});};timingLocations.append(button);
  }
}
function generationOrigin(candidate){
  let current=candidate;const visited=new Set();
  while(current&&typeof current==='object'){
    const kind=current.generation_origin?.kind;
    if(kind==='regional_generation')return 'regional';
    if(kind==='full_song_generation')return 'full_song';
    if(current.region!=null||current.selection?.scope==='regional_output')return 'regional';
    if(kind==='unknown')return 'unknown';
    const edits=current.edits;
    if(edits===undefined||edits===null||Array.isArray(edits)&&!edits.length)return 'full_song';
    if(!Array.isArray(edits))return 'unknown';
    const parentCreated=edits.at(-1)?.parent_created;
    if(!isCreatedKey(parentCreated)||visited.has(parentCreated))return 'unknown';
    visited.add(parentCreated);
    const parents=(state?.variants||[]).filter(item=>isCreatedKey(item.created)&&item.created===parentCreated);
    if(parents.length!==1)return 'unknown';
    current=parents[0];
  }
  return 'unknown';
}
function isCreatedKey(value){return typeof value==='string'||typeof value==='number'&&Number.isFinite(value);}
function loadState(s,{newAudioProject=false}={}){
  projectLoadRequest++;const changed=state?.id!==s.id||state?.original!==s.original;cancelAuditionPlayback();cueClock.reset();state=s;
  setSettings(newAudioProject&&s.mode==='audio'&&(!Array.isArray(s.variants)||s.variants.length===0)
    ?{...s.settings,...GENERATION_PRESETS.hybrid_adaptive}
    :s.settings);
  $('projectTitle').textContent=s.title;$('projectSub').textContent=s.mode==='remap'?`已导入 ${s.input_keys}K 原谱 · ${s.input_notes.length} 个物件`:'本地音频项目 · 先检查采音，再调整编排';
  $('mode').value=s.mode;
  if(s.mode==='remap')$('singleEngine').value='traditional';
  updateSingleEngineControls();
  options($('charts'),s.charts.map(c=>[c.path,c.label]));$('charts').classList.toggle('hidden',!s.charts.length);$('charts').value=s.original||'';
  const v=variant();if(v)$('mode').value=v.mode;if(s.analysis)$('analysisEngine').value=s.analysis.separation?(s.analysis.separation.source==='provided'?'provided':s.analysis.separation.profile==='six'?'deep6':s.analysis.separation.profile==='four-piano'?'deep4piano':'deep'):'basic';updateVoiceOptions(true);
  lastClockShift=v?.audio_shift_ms??s.audio_shift_ms??(s.analysis?0:22.5);$('audioShift').value=lastClockShift;
  const tempo=v||s.analysis?.tempo;$('bpm').value=tempo?.bpm??120;$('offset').value=autoChartOffset((tempo?.offset??0)-(v||!s.analysis?0:clockShift()));
  clockAnchor=!v&&s.analysis?s.analysis.tempo.offset:Number($('offset').value)+clockShift();
  if(changed)$('auditionOffset').value=$('offset').value;$('timingConfirmed').checked=!!v?.timing_confirmed;$('division').value=tempo?.division||4;
  tempoChoices=[...(s.analysis?.tempo?.candidates||[]),...(s.analysis?.tempo?.precise_candidates||[])].map(c=>({...c,audio_offset:c.offset,offset:autoChartOffset(c.offset-clockShift())}));showTempoChoices();
  renderOriginalTempoNote();clearTimingDiagnostics();
  if(v?.timing_diagnostics)showTimingDiagnostics(v.timing_diagnostics);
  else if(v?.mode==='audio')$('timingReport').textContent='当前拍线尚未检查；旧候选未记录完整诊断，请重新检查整曲。';
  renderTimingLocations();
  $('planText').textContent=s.plan?.explanation||'';
  const ref=s.reference_style;$('referenceStyleInfo').textContent=ref?`${ref.title} · ${ref.keys}K · LN ${(ref.ln_ratio*100).toFixed(1)}% · 中位长度 ${ref.ln_duration_ms['50']||0} ms · 峰值 ${ref.peak_nps} 物件/秒。${ref.warnings.join(' ')} ${ref.notice}`:'不复制原谱音符或时间，也不保证相同手感。';
  options($('variants'),s.variants.length?s.variants.map((v,i)=>{const origin=generationOrigin(v);return [i,`候选 ${i+1} · ${v.settings.keys}K · seed ${v.settings.seed} · ${v.notes.length} 物件${v.region?' · 局部':origin==='regional'?' · 编辑自局部候选':origin==='unknown'?' · 来源未知':''}${v.edits?.length?' · 已修正 '+v.edits.length+' 次':''}`];}):[['','尚未生成']]);$('variants').value=s.active;
  setPlaybackSource(changed);updateAudioStatus();
  [['exportOsu','osu'],['exportOsz','osz'],['exportJson','json']].forEach(([id,kind])=>{const a=$(id);a.title=kind==='osz'&&!s.audio?'请导入配套音频后导出谱包':'';a.download='ManiaLab.'+kind;});
  const report=v?.report;const vals=report?[report.notes,`${(report.ln_ratio*100).toFixed(1)}%`,report.max_held,`${(report.longest_ln_ms/1000).toFixed(2)}s`]:['—','—','—','—'];document.querySelectorAll('.metrics strong').forEach((e,i)=>e.textContent=vals[i]);
  selected=null;$('inspector').textContent='点击音符查看起止时间与来源。';window.resetWaveDetail?.();setMode();updateGridStatus();renderReview();window.renderLNDesign?.();window.renderNoteEditor?.();window.renderRecommendations?.();window.renderStructure?.();window.renderProvidedStems?.();window.renderPianoDetector?.();window.renderInstrumentDetector?.();window.renderGuitarDetector?.();draw();updateChartPreview();refreshRecent();
}
function updateGridStatus(){
  const v=variant(),audioMode=v?.mode==='audio';
  const candidateLayout=v?.settings.key_layout||DEFAULT_KEY_LAYOUT[v?.settings.keys];
  const keyMismatch=!!v&&(keys!==v.settings.keys||$('keyLayout').value!==candidateLayout);
  const settingsMismatch=!!v&&candidateSettingsMismatch(v);
  const mismatch=audioMode&&(['bpm','offset','division'].some(k=>Number($(k).value)!==v[k])||clockShift()!==(v.audio_shift_ms||0));
  const stale=audioMode&&(v.timing_revision||0)<3;
  const g=v?.report?.grid;
  const diagnosis=$('exportJson'),hasDiagnosis=!!(v||state?.analysis);
  diagnosis.classList.toggle('disabled',!hasDiagnosis);if(hasDiagnosis)diagnosis.href=candidateExportUrl('json',v);else diagnosis.removeAttribute('href');
  const draftStatus=$('candidateDraftStatus'),actualLn=v?.report?.ln,actualRatio=v?.report?.ln_ratio;
  const presetLabel=$('preset').selectedOptions[0]?.textContent||'自定义';
  draftStatus.classList.toggle('hidden',!settingsMismatch);
  draftStatus.textContent=settingsMismatch
    ?`当前候选实际含 ${actualLn??'未知'} 个 LN${Number.isFinite(actualRatio)?`（${(actualRatio*100).toFixed(1)}%）`:''}。当前「${presetLabel}」只用于下一次生成；请重新生成后导出。`
    :'';
  $('gridReport').textContent=keyMismatch?(keys!==v.settings.keys?`目标为 ${keys}K，当前预览仍是 ${v.settings.keys}K 候选`:`目标键型 ${$('keyLayout').value} 与当前预览键型 ${candidateLayout} 不同`)+'；请重新生成后导出。':settingsMismatch?'生成预设或参数与当前候选不同；当前候选未自动修改。请重新生成后导出。':stale?'旧版候选：请重新分析并生成整谱，才能导出。':mismatch?'当前 BPM / Offset / 细分与预览候选不同，请重新生成后导出。':g?`落线检查：${g.endpoints-g.off_grid} / ${g.endpoints} 个短键起点与 LN 起止在网格上；最大舍入误差 ${g.max_error_ms} ms。落线检查不判断采音是否正确。`:'';
  ['exportOsu','exportOsz'].forEach(id=>{const a=$(id),blocked=!v||v.report?.valid===false||stale||mismatch||keyMismatch||settingsMismatch||(id==='exportOsz'&&!state?.audio);a.classList.toggle('disabled',blocked);if(blocked)a.removeAttribute('href');else a.href=candidateExportUrl(id==='exportOsz'?'osz':'osu',v);});
}
function updateChartPreview(){
  const root=$('chartPreviewHost');if(!root||!previewController)return;
  const v=variant(),notes=v?.notes||state?.input_notes||[];
  const stats={
    objects:v?.report?.notes??notes.length,
    ln_ratio:v?.report?.ln_ratio,
    peak_held:v?.report?.max_held,
    hold_notes:v?.report?.ln,
    torosu:v?.report?.v32?.torosu||v?.report?.torosu||v?.selection?.torosu||{},
  };
  previewController.setChart({
    notes:notes.map(note=>({start_ms:note.start,end_ms:note.end,lane:note.lane,is_hold:note.hold||note.end>note.start})),
    keys:v?.settings?.keys||state?.input_keys||keys,
    bpm:v?.bpm||120,offset:v?.offset||0,division:v?.division||4,meter:v?.meter||4,stats,
  });
}
function v32StatusText(engine){
  if(!engine)return 'V32 引擎状态尚未报告。';
  if(typeof engine==='string')return engine;
  const device=engine.device||{},deviceType=typeof device==='string'?device:device.type||'未知';
  const deviceParts=[deviceType.toUpperCase()];
  if(device.name)deviceParts.push(device.name);
  if(device.memory_gib!==undefined&&device.memory_gib!==null&&Number.isFinite(Number(device.memory_gib)))deviceParts.push(`${Number(device.memory_gib)} GiB`);
  if(device.precision)deviceParts.push(device.precision);
  if(device.fast_decoder_loop)deviceParts.push('fast decoder');
  const modelText=key=>{
    const model=engine.models?.[key];
    if(!model)return '未报告';
    const ready=typeof model==='object'?model.ready:model;
    const revision=typeof model==='object'&&model.revision?` · ${String(model.revision).slice(0,12)}`:'';
    return `${ready?'已就绪':'待下载'}${revision}`;
  };
  const lora=engine.lora||{};
  const loraText=lora.loaded?`已加载${lora.path?` · ${lora.path}`:''}`:'未加载';
  return [
    `设备：${deviceParts.filter(Boolean).join(' · ')}`,
    `引擎：${engine.available?'可用':'不可用'} · 源码 ${engine.source_available?'就绪':'缺失'} · 运行时 ${engine.runtime_available?'就绪':'缺失'}${engine.device_override_error?` · ${engine.device_override_error}`:''}`,
    `权重：V32 ${modelText('v32')} · V32-mini ${modelText('v32-mini')}`,
    `LN RMS：${engine.separation_available===false?'使用原混音 RMS fallback（轻量工作台）':'使用六声部分轨 RMS'}`,
    `LoRA：${loraText}`,
  ].join('\n');
}
function updateV32Outputs(){
  $('v32HoldRatioOut').textContent=`${Math.round(Number($('v32HoldRatio').value)*100)}%`;
  $('v32TemperatureOut').textContent=Number($('v32Temperature').value).toFixed(1);
}
async function refreshRecent(){
  const status=await api('/api/status'),profiles=status.separator?.profiles||{};
  try{
    const registry=await api('/api/presets');
    for(const item of registry.items||[])if(item?.id&&item.settings){
      GENERATION_PRESETS[item.id]=item.settings;
      if(item.v32?.hold_note_ratio!==undefined)PRESET_V32_HOLD[item.id]=item.v32.hold_note_ratio;
      if(item.torosu?.density)PRESET_TOROSU_DEFAULTS[item.id]=item.torosu.density;
    }
  }catch(_error){}
  separatorAvailable=!!status.separator?.available;
  const pianoFourAvailable=!!profiles['four-piano']?.available;
  $('analysisEngine').querySelector('[value=deep]').disabled=!separatorAvailable;
  $('analysisEngine').querySelector('[value=deep4piano]').disabled=!pianoFourAvailable;
  $('analysisEngine').querySelector('[value=deep6]').disabled=!profiles.six?.available;
  $('separationStatus').textContent=pianoFourAvailable?'主声部使用鼓、钢琴、人声、其他伴奏；完整六声部另含贝斯与吉他。批量分析会沿用所选声部方案。':separatorAvailable?'当前包有兼容四声部（鼓、贝斯、人声、其他伴奏）；可使用快速采音':'当前包未包含分离组件 · 可使用快速采音';
  if(!state)$('analysisEngine').value=pianoFourAvailable?'deep4piano':separatorAvailable?'deep':'basic';
  window.updatePianoStatus?.(status.piano);window.updateInstrumentStatus?.(status.instruments);window.updateGuitarStatus?.(status.guitar);
  options($('recent'),[['','打开已保存项目…'],...(status.projects||[]).map(p=>[p.id,p.title])]);
  if(state)$('recent').value=state.id;
  mapperInstallAvailable=!!status.engine?.install_available;mapperGenerateAvailable=!!status.engine?.generate_available;
  $('engineInstall').disabled=busy||!mapperInstallAvailable;$('engineGenerate').disabled=busy||!mapperGenerateAvailable;
  $('engineStatus').textContent=status.engine?.installed?`运行环境已就绪 · ${status.engine.model_downloaded?'模型权重已下载':'模型权重将在首次生成时下载'}`:mapperInstallAvailable?'原模型未安装 · 不影响基础采音与定轨':'当前便携配置不含原模型安装器 · 不影响基础采音与定轨';
  v32EngineStatus=status.engine?.v32_engine??status.v32_engine??null;
  v32GenerateAvailable=!!v32EngineStatus?.available;
  if(!singleEngineTouched){
    $('singleEngine').value=v32GenerateAvailable?'v32-mini':'traditional';
    updateSingleEngineControls();
  }
  if(!batchEngineTouched)$('batchEngine').value=v32GenerateAvailable?'v32-mini':'traditional';
  $('v32Status').textContent=v32StatusText(v32EngineStatus);
  const device=v32EngineStatus?.device||{},deviceType=typeof device==='string'?device:device.type;
  $('hardwareStatus').textContent=deviceType==='cuda'?`CUDA GPU 加速 · ${device.name||'NVIDIA'}`:'纯 CPU 模式';
  $('v32ReadyStatus').textContent=v32EngineStatus?.available
    ?`V32 就绪 · ${v32EngineStatus.models?.['v32-mini']?.ready?'mini 权重已缓存':'首次生成下载权重'}${v32EngineStatus.separation_available===false?' · 混音 RMS fallback':''}`
    :`V32 ${v32EngineStatus?.runtime_available?'环境待配置':'未就绪'}`;
  $('torosuReadyStatus').textContent=status.torosu?.ready?'Torosu 已就绪':'Torosu 未就绪';
  const v32CanInstall=!!v32EngineStatus?.install_available&&(!v32EngineStatus?.runtime_available||v32EngineStatus?.runtime_needs_cuda);
  $('v32Install').classList.toggle('hidden',!v32CanInstall);
  $('v32Install').textContent=v32EngineStatus?.runtime_needs_cuda?'配置 NVIDIA CUDA 环境':'准备 V32 运行环境';
  $('v32Install').disabled=busy||!v32CanInstall;
  $('v32MirrorOption').classList.toggle('hidden',!v32CanInstall);
  $('v32UseMirrors').disabled=busy||!v32CanInstall;
  $('headerV32Install').classList.toggle('hidden',!v32CanInstall);
  $('headerV32Install').disabled=busy||!v32CanInstall;
  $('v32Generate').disabled=busy||!state||$('mode').value==='remap'||!v32GenerateAvailable;
  $('priorStatus').textContent=status.pattern_prior?.available?`通过验证的键数：${status.pattern_prior.keys.map(k=>k+'K').join(' / ')}。仅提供定轨软偏好；未通过的键数自动忽略。`:'当前未加载经过验证的键型先验。';
  const loraResponse=await api('/api/loras').catch(()=>({items:[]}));
  localLoras=loraResponse.items||[];
  const loraOptions=[['','不挂载 LoRA'],...localLoras.map(item=>[item.path,`${item.label}${item.base_model?` · ${item.base_model}`:''}`])];
  options($('v32LoraSelect'),loraOptions);options($('batchLora'),loraOptions);
  await refreshBatchHistory();
}
function setMode(){const remap=$('mode').value==='remap';$('musical').classList.toggle('hidden',remap);$('strictHint').classList.toggle('hidden',!remap);$('bpm').disabled=remap||busy;$('offset').disabled=remap||busy;['tempoCandidates','halfBpm','doubleBpm','division','timingConfirmed','checkTiming','refitTiming','referenceTiming','applyTiming','audioShift','waveAlign','auditionOffset','auditionB','auditionApply'].forEach(id=>$(id).disabled=remap||busy);$('auditionA').textContent=remap?'试听原谱拍线':'试听当前拍线 A';$('generate').firstChild.textContent=remap?'严格重新定轨 ':'生成候选谱面 ';$('v32Generate').disabled=busy||!state||remap||!v32GenerateAvailable;}
// A pending recent-project GET must not replace the project being acted on.
function setBusy(value){
  if(value)projectLoadRequest++;
  busy=value;
  ['analyze','generate','another','regenerate','plan','charts','variants','recent','files','batchFiles','batchAnalyze','batchKeys','batchKeyLayout','batchPreset','batchTorosuDensity','batchTorosuLnMode','batchDensity','batchEngine','batchDifficulty','batchHoldRatio','batchLora','batchLoraPath','batchScribe','batchVersionSuffix','batchTimingConfirmed','batchGenerate','batchDownloadAll','clearPlan','engineInstall','engineGenerate','v32Install','v32UseMirrors','v32Model','v32Strategy','v32Density','v32Difficulty','v32HoldRatio','v32Temperature','v32LnMode','v32LoraSelect','v32LoraPath','v32Generate','headerV32Install','singleEngine','singleScribe','singleVersionSuffix','referenceStyleFile','mode','preset','lnPreset','focus','layoutEngine','analysisEngine',...numbers].forEach(id=>{const control=$(id);if(control)control.disabled=value;});
  $('engineInstall').disabled=value||!mapperInstallAvailable;$('engineGenerate').disabled=value||!mapperGenerateAvailable;
  $('v32Install').disabled=value||!v32EngineStatus?.install_available||(
    !!v32EngineStatus?.runtime_available&&!v32EngineStatus?.runtime_needs_cuda
  );
  $('headerV32Install').disabled=value||!v32EngineStatus?.install_available||(
    !!v32EngineStatus?.runtime_available&&!v32EngineStatus?.runtime_needs_cuda
  );
  $('v32Generate').disabled=value||!state||$('mode').value==='remap'||!v32GenerateAvailable;
  document.querySelectorAll('[data-key]').forEach(button=>button.disabled=value);
  $('job').classList.toggle('hidden',!value);setMode();window.updateEditorLocks?.();window.lockStructure?.();window.lockProvidedStems?.();window.lockPianoDetector?.();window.lockInstrumentDetector?.();window.lockGuitarDetector?.();
  document.querySelectorAll('[data-recommendation],#recommendationRefresh').forEach(button=>button.disabled=value||!state?.analysis);
  updateBatchAvailability();if(batchSnapshot)renderBatchStatus(batchSnapshot);else updateBatchControls();
}
async function runJob(action,payload,form=false){
  if(!state)throw Error('请先导入素材');if(busy)return;const projectId=state.id;setBusy(true);
  try{const job=await api(`/api/projects/${projectId}/${action}`,payload,form);jobId=job.id;
    while(true){const j=await api('/api/jobs/'+jobId);$('progress').value=j.progress;$('jobText').textContent=j.message;
      if(['done','error','cancelled'].includes(j.status)){if(j.status==='error')throw Error(j.message);if(j.status==='done'){loadState(await api('/api/projects/'+projectId));const v=variant();msg(action==='provided-stems'?'独立声轨已保存。点击分析音频，逐乐器检查采音与音乐分段。':action==='analyze'?`已提取 ${state.analysis.candidates.length} 个采音候选。切换「采音证据」试听检查。`:action==='structure'?`已划分 ${state.analysis.structure.segments.length} 个音乐段落，请检查声部表达方案。`:action==='plan'?'编排建议已应用，点击生成即可试谱。':action==='engine-install'?'原模型运行环境已安装。首次生成将自动下载 mania 权重。':action==='v32-install'?'V32 独立运行环境已准备完成；已自动选择 CUDA 或 CPU。':action==='v32-generate'?v?`V32 候选已保存到项目历史（${v.settings?.keys??10}K · ${v.report?.notes??v.notes?.length??0} 个物件）。请检查当前候选并使用 .osu/.osz 导出。`:'V32 任务完成，项目已刷新。':`${v.report.notes} 个物件，${v.report.ln} 个 LN；${v.report.valid?'结构检查通过':`发现 ${v.report.issue_count} 项结构冲突，请先检查`}。${v.selection.message||''}\n编排耗时 ${j.elapsed_seconds??v.elapsed_seconds??'未知'} 秒。${v.selection.feasible_ln_ratio!==undefined?` 可达 LN 比例${v.selection.maximum_proven?'上限':'（已求得）'} ${(v.selection.feasible_ln_ratio*100).toFixed(1)}%。`:''}`);if(j.cleanup_failed)msg($('message').textContent+' '+j.message,true);}else msg(j.message);return j.status;}
      await new Promise(r=>setTimeout(r,550));
    }
  }finally{jobId=null;setBusy(false);}
}
async function runGlobalV32Install(){
  if(busy)return;
  if(!v32EngineStatus?.install_available)throw Error('V32 运行环境当前无需安装。');
  setBusy(true);$('jobText').textContent='正在准备 V32 全局运行环境…';$('progress').value=0;
  try{
    const job=await api('/api/v32/install',{use_mirrors:$('v32UseMirrors').checked});
    jobId=job.id;
    while(true){
      const current=await api('/api/jobs/'+encodeURIComponent(jobId));
      $('progress').value=current.progress;$('jobText').textContent=current.message;
      if(['done','error','cancelled'].includes(current.status)){
        if(current.status==='error')throw Error(current.message);
        await refreshRecent();
        msg(current.status==='done'?'V32 全局运行环境已准备完成；新项目可直接选择 V32。':current.message);
        return current.status;
      }
      await new Promise(resolve=>setTimeout(resolve,550));
    }
  }finally{jobId=null;setBusy(false);}
}
function analysisRequestOptions({batch=false}={}){
  const engine=$('analysisEngine').value;
  if(batch&&!['basic','deep','deep6','deep4piano'].includes(engine))throw Error('批量分析仅支持快速、四声部、主声部钢琴或完整六声部音频分析。');
  return {settings:settings(),engine,piano_engine:window.pianoSelection?.()||'general',...(window.instrumentSelection?.()||{instrument_engine:'general',allow_predicted_releases:false}),...(window.guitarSelection?.()||{guitar_engine:'general',allow_guitar_predicted_releases:false}),audio_shift_ms:clockShift()};
}
function updateBatchAvailability(){
  const input=$('batchFiles'),button=$('batchAnalyze'),summary=$('batchFileSummary');
  if(!input||!button||!summary)return;
  const files=[...input.files],count=files.length;
  const totalBytes=files.reduce((sum,file)=>sum+file.size,0),maxBytes=20*512*1024*1024;
  button.disabled=busy||count<2||count>20||totalBytes>maxBytes;
  if(count===0)summary.textContent='尚未选择歌曲。';
  else if(count===1)summary.textContent='已选择 1 首；批量分析至少需要 2 首。';
  else if(count>20)summary.textContent='已选择 '+count+' 首；最多处理 20 首，请减少选择。';
  else if(totalBytes>maxBytes)summary.textContent='总文件大小超过 10 GiB，请减少选择。';
  else summary.textContent='已选择 '+count+' 首 · '+(totalBytes/1024/1024).toFixed(1)+' MB · 最多同时分析 2 首。';
}
function batchReadyItems(){
  return (batchAnalysisSnapshot?.items||[]).filter(item=>item.status==='done'&&(item.project||item.project_id));
}
function batchCandidateId(item){return item?.result?.candidate_id||item?.candidate_id||'';}
function batchProjectId(item){return item?.project||item?.project_id;}
function batchGenerationRows(items){
  const rows=Array.isArray(items)?items:[];
  return batchGenerationAttemptIds.size
    ?rows.filter(item=>batchGenerationAttemptIds.has(batchProjectId(item)))
    :rows;
}
function mergeBatchGenerationStatus(snapshot){
  const generated=new Map((snapshot.items||[]).map(item=>[batchProjectId(item),item]));
  const items=(batchAnalysisSnapshot?.items||[]).map(analysisItem=>{
    const project=batchProjectId(analysisItem);
    return generated.get(project)||analysisItem;
  });
  return {...snapshot,generation_total:snapshot.generation_total||(snapshot.items||[]).length,
    generation_failed:(snapshot.items||[]).filter(item=>item.status==='error').length,items};
}
function updateBatchKeyLayout(){
  const requested=$('batchKeyLayout').value;
  options($('batchKeyLayout'),KEY_LAYOUT_CHOICES[Number($('batchKeys').value)]||[]);
  const choices=KEY_LAYOUT_CHOICES[Number($('batchKeys').value)]||[];
  $('batchKeyLayout').value=choices.some(([value])=>value===requested)?requested:
    DEFAULT_KEY_LAYOUT[Number($('batchKeys').value)]||choices[0]?.[0]||'';
}
function updateBatchControls(){
  const controls=$('batchGenerateControls');if(!controls)return;
  const ready=batchReadyItems();
  const generationItems=batchPhase==='generation'?batchSnapshot?.items||[]:[];
  const generated=generationItems.filter(item=>item.status==='done'&&batchCandidateId(item));
  controls.classList.toggle('hidden',!ready.length&&!generated.length);
  $('batchGenerate').disabled=busy||!ready.length||batchResultsOnly;
  $('batchDownloadAll').classList.toggle('hidden',!generated.length);
  const downloadDisabled=busy||!generated.length;
  $('batchDownloadAll').setAttribute('aria-disabled',String(downloadDisabled));
  $('batchDownloadAll').classList.toggle('disabled',downloadDisabled);
  if(generated.length&&!busy){
    const encoded=generated.map(item=>`${item.project||item.project_id}:${batchCandidateId(item)}`).join(',');
    $('batchDownloadAll').href='/api/batch/export-osz?'+new URLSearchParams({items:encoded}).toString();
    $('batchDownloadAll').removeAttribute('tabindex');
  }else{
    $('batchDownloadAll').removeAttribute('href');
    $('batchDownloadAll').setAttribute('tabindex','-1');
  }
  $('batchDensityOut').textContent=$('batchDensity').value;
}
async function batchUnstableTimingTitles(items){
  const projects=await Promise.all(items.map(item=>api('/api/projects/'+encodeURIComponent(item.project||item.project_id))));
  return projects.filter(project=>!project.analysis?.tempo?.stable).map(project=>project.title||project.id);
}
function batchDownloadName(title){
  const safe=String(title||'ManiaLab').replace(/[<>:"/\\|?*\u0000-\u001f]/g,'_').trim().slice(0,100);
  return (safe||'ManiaLab')+'.osz';
}
let reconnectingBatch=false;
async function refreshBatchHistory(){
  const root=$('batchHistory');if(!root)return;
  try{
    const {items=[]}=await api('/api/batch/jobs?limit=12');
    root.replaceChildren();
    if(!items.length){const empty=document.createElement('p');empty.className='muted compact';empty.textContent='暂无批量任务记录。';root.append(empty);return;}
    const title=document.createElement('p');title.className='batch-subheading';title.textContent='本机批次记录';root.append(title);
    for(const item of items){
      const row=document.createElement('div');row.className='batch-history-item';
      const copy=document.createElement('div');copy.className='batch-item-copy';
      const name=document.createElement('strong');name.textContent=`${item.resume_kind==='analysis'?'批量分析':'批量生成'} · ${(item.items||[]).length} 首`;
      const status=document.createElement('small');status.textContent=`${item.status==='interrupted'?'已中断':item.status==='done'?'已完成':item.status==='error'?'有失败':item.status==='running'?'运行中':'已取消'} · ${new Date((item.created_at||0)*1000).toLocaleString()}`;
      copy.append(name,status);row.append(copy);
      if(item.reconnect_available){
        const watch=document.createElement('button');watch.type='button';watch.dataset.watchBatch=item.id;watch.textContent='恢复进度';row.append(watch);
      }
      if(item.resume_available){
        const resume=document.createElement('button');resume.type='button';resume.dataset.resumeBatch=item.id;resume.textContent='续跑未完成';row.append(resume);
      }
      if(item.view_results_available&&item.resume_kind!=='analysis'){
        const results=document.createElement('button');results.type='button';
        results.dataset.showBatchResults=item.id;results.textContent='恢复下载列表';row.append(results);
      }
      root.append(row);
    }
    const active=items.find(item=>item.reconnect_available);
    if(active&&!busy&&!reconnectingBatch)await reconnectPersistentBatch(active.id);
  }catch(error){
    root.replaceChildren();
    const note=document.createElement('p');note.className='muted compact';note.textContent='批次记录暂不可用。';root.append(note);
  }
}
async function reconnectPersistentBatch(jobIdToWatch){
  if(busy||reconnectingBatch)return;
  batchResultsOnly=false;
  reconnectingBatch=true;
  const initial=await api(`/api/batch/jobs/${encodeURIComponent(jobIdToWatch)}`).catch(()=>null);
  if(!initial){reconnectingBatch=false;return;}
  const analysis=initial.resume_kind==='analysis';
  batchPhase=analysis?'analysis':'generation';
  if(analysis){
    batchImported=(initial.items||[]).map(item=>({id:item.project,title:item.title}));
    batchAnalysisSnapshot=null;
  }else{
    batchAnalysisSnapshot={items:(initial.items||[]).map(item=>({
      ...item,
      status:item.status==='done'?'done':item.status==='error'?'error':'interrupted',
      ...(batchCandidateId(item)?{result:item.result||{candidate_id:batchCandidateId(item)}}:{}),
    }))};
  }
  readyBatchProjectIds=new Set((initial.items||[]).map(item=>item.project||item.project_id));
  batchGenerationAttemptIds=new Set(
    analysis?[]:(initial.items||[]).map(batchProjectId).filter(Boolean),
  );
  batchAttemptJobId=analysis?null:jobIdToWatch;
  setBusy(true);batchJobId=jobIdToWatch;jobId=jobIdToWatch;
  try{
    let current=initial;
    renderBatchStatus(analysis?current:mergeBatchGenerationStatus(current));
    while(!['done','error','cancelled','interrupted'].includes(current.status)){
      await new Promise(resolve=>setTimeout(resolve,650));
      current=await api(`/api/jobs/${encodeURIComponent(jobIdToWatch)}`);
      $('progress').value=current.progress||0;
      $('jobText').textContent=current.message||'正在恢复批量任务…';
      renderBatchStatus(analysis?current:mergeBatchGenerationStatus(current));
    }
    if(analysis){
      batchAnalysisSnapshot=current;batchPhase='analysis';
    }
    msg(`已恢复${analysis?'分析':'生成'}进度：${(current.items||[]).filter(item=>item.status==='done').length}/${current.items?.length||0} 首完成。`);
  }catch(error){
    msg(`无法恢复批次进度：${error.message}`,true);
  }finally{
    batchJobId=null;jobId=null;setBusy(false);reconnectingBatch=false;
    await refreshBatchHistory();
  }
}
async function resumePersistentBatch(jobIdToResume){
  if(busy)return;
  batchResultsOnly=false;
  const previous=await api(`/api/batch/jobs/${encodeURIComponent(jobIdToResume)}`);
  if(!previous.resume_available)throw Error('此批次没有未完成歌曲可续跑。');
  const analysis=previous.resume_kind==='analysis';
  const completedBefore=(previous.items||[]).filter(item=>item.status==='done');
  batchPhase=analysis?'analysis':'generation';
  batchAnalysisSnapshot=analysis?null:{items:(previous.items||[]).map(item=>({
    ...item,
    status:item.status==='done'?'done':item.status==='error'?'error':'interrupted',
    ...(batchCandidateId(item)?{result:item.result||{candidate_id:batchCandidateId(item)}}:{}),
  }))};
  readyBatchProjectIds=new Set((previous.items||[]).map(item=>item.project||item.project_id));
  batchGenerationAttemptIds=new Set(
    analysis?[]:(previous.items||[])
      .filter(item=>item.status!=='done')
      .map(batchProjectId)
      .filter(Boolean),
  );
  batchAttemptJobId=null;
  batchSnapshot=previous;batchCancelRequested=false;batchGenerationSubmitting=true;
  setBusy(true);$('jobText').textContent='正在恢复批量任务…';$('progress').value=0;
  try{
    const queued=await api(`/api/batch/jobs/${encodeURIComponent(jobIdToResume)}/resume`,{});
    batchJobId=queued.id;jobId=queued.id;
    if(!analysis){
      batchGenerationAttemptIds=new Set((queued.items||[]).map(batchProjectId).filter(Boolean));
      batchAttemptJobId=queued.id;
      return await monitorBatchGeneration(queued,queued.items.length);
    }
    let current=queued;
    while(!['done','error','cancelled'].includes(current.status)){
      await new Promise(resolve=>setTimeout(resolve,650));
      current=await api('/api/jobs/'+encodeURIComponent(queued.id));
      $('progress').value=current.progress||0;
      $('jobText').textContent=current.message||'正在续跑批量分析…';
      renderBatchStatus(current);
    }
    const merged=new Map((current.items||[]).map(item=>[item.project||item.project_id,item]));
    for(const item of completedBefore)merged.set(item.project||item.project_id,item);
    batchAnalysisSnapshot={...current,items:[...merged.values()]};batchPhase='analysis';
    renderBatchStatus(batchAnalysisSnapshot);
    msg(`批量分析续跑结束：${(batchAnalysisSnapshot.items||[]).filter(item=>item.status==='done').length}/${batchAnalysisSnapshot.items?.length||0} 首成功。`,current.status==='error');
    await refreshRecent();
    return current.status;
  }finally{
    batchGenerationSubmitting=false;batchJobId=null;jobId=null;batchCancelRequested=false;setBusy(false);
    await refreshBatchHistory();
  }
}
function setWorkflowMode(mode){
  const batch=mode==='batch';
  document.querySelectorAll('[data-work-mode]').forEach(button=>{
    const active=button.dataset.workMode===mode;
    button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));
  });
  document.querySelectorAll('[data-single-workflow]').forEach(element=>element.classList.toggle('hidden',batch));
  const panel=document.querySelector('.batch-panel');
  panel.hidden=!batch;
  if(batch)panel.open=true;
  if(!batch)panel.open=false;
  $('batchFiles').disabled=busy;
}
function selectedLora(){
  const selected=$('v32LoraSelect')?.value||$('batchLora')?.value||'';
  return selected||$('v32LoraPath')?.value?.trim()||$('batchLoraPath')?.value?.trim()||'';
}
function applyGenerationPreset(id){
  const preset=GENERATION_PRESETS[id];if(!preset)return;
  setSettings({...settings(),...preset});
  if(id==='torosu_remap'){$('mode').value='remap';$('singleEngine').value='traditional';}
  else {if($('mode').value==='remap')$('mode').value='audio';if(PRESET_V32_HOLD[id]!==undefined)$('v32HoldRatio').value=PRESET_V32_HOLD[id];}
  $('batchTorosuDensity').value=PRESET_TOROSU_DEFAULTS[id]||'dense';
  $('batchTorosuLnMode').value=preset.torosu_ln_mode||'keep';
  $('batchHoldRatio').value=PRESET_V32_HOLD[id]??.3;
  $('batchHoldRatioOut').textContent=`${Math.round(Number($('batchHoldRatio').value)*100)}%`;
  updateV32Outputs();
  updateSingleEngineControls();
  setMode();updateGridStatus();
}
function updateSingleEngineControls(){
  const engine=$('singleEngine').value;
  $('v32Model').value=engine==='v32'?'v32':'v32-mini';
  $('v32PanelControls').classList.toggle('hidden',engine==='traditional');
}
function renderBatchStatus(snapshot){
  batchSnapshot=snapshot;
  const root=$('batchStatus');if(!root)return;
  const items=Array.isArray(snapshot.items)?snapshot.items:[];
  const done=items.filter(item=>item.status==='done').length;
  const failed=items.filter(item=>['error','cancelled'].includes(item.status)).length;
  const generationRows=batchGenerationRows(items);
  const generationDone=generationRows.filter(item=>item.status==='done').length;
  const generationFailed=generationRows.filter(item=>item.status==='error').length;
  const heading=document.createElement('p');heading.className='batch-heading';
  if(snapshot.status==='importing')heading.textContent=snapshot.message||'正在逐首导入音频…';
  else if(snapshot.status==='queued')heading.textContent=batchPhase==='generation'?'统一预设已提交，等待生成…':'批量分析已排队，等待处理…';
  else if(snapshot.status==='running'){
    heading.textContent=batchPhase==='generation'
      ?`正在统一生成：${generationDone}/${snapshot.generation_total||generationRows.length} 首完成${generationRows.some(item=>item.status==='error')?'，有歌曲失败':''}。`
      :'正在分析：'+done+'/'+items.length+' 首完成'+(failed?'，'+failed+' 首失败':'')+'。';
  }
  else if(snapshot.status==='done')heading.textContent=batchPhase==='generation'
    ?'统一生成结束：'+generationDone+'/'+(snapshot.generation_total||generationRows.length)+' 首成功'+((snapshot.generation_failed??generationFailed)?'，'+(snapshot.generation_failed??generationFailed)+' 首生成失败':'')+
      (failed>snapshot.generation_failed?'；分析失败的歌曲仍保留':'')+'。'
    :'批次完成：'+done+'/'+items.length+' 首成功'+(failed?'，'+failed+' 首失败':'')+'。';
  else if(snapshot.status==='cancelled')heading.textContent='批次已取消：'+done+'/'+items.length+' 首完成。';
  else heading.textContent=snapshot.message||'批次结束：'+done+'/'+items.length+' 首成功。';
  const list=document.createElement('div');list.className='batch-items';
  for(const item of items){
    const row=document.createElement('div');row.className='batch-item';row.dataset.batchItem='';
    const copy=document.createElement('div');copy.className='batch-item-copy';
    const title=document.createElement('strong');title.textContent=item.title||item.name||'未命名歌曲';
    const status=document.createElement('span');status.className='batch-item-state';
    const statusText={queued:'等待中',running:(batchPhase==='generation'?'生成中 ':'分析中 ')+Math.round(item.progress||0)+'%',done:batchPhase==='generation'?'生成完成':'分析完成',error:'失败',cancelled:'已取消',interrupted:'中断 · 可续跑',imported:'已导入，未分析'};
    status.textContent=statusText[item.status]||item.message||'等待中';
    const detail=document.createElement('small');detail.textContent=item.status==='error'?(item.error||item.message||(batchPhase==='generation'?'生成失败':'分析失败')):item.message||'';
    copy.append(title,status);if(detail.textContent)copy.append(detail);
    row.append(copy);
    const projectId=item.project||item.project_id;
    if(projectId){
      const open=document.createElement('button');open.type='button';open.className='quiet';open.dataset.openProject=projectId;open.textContent='打开项目';open.disabled=busy;
      row.append(open);
      const candidateId=batchPhase==='generation'&&item.status==='done'?batchCandidateId(item):'';
      if(candidateId){
        const download=document.createElement('a');download.className='batch-download-link';download.dataset.batchDownload='';
        if(busy){download.classList.add('disabled');download.setAttribute('aria-disabled','true');download.setAttribute('tabindex','-1');}
        else{download.href=`/api/projects/${encodeURIComponent(projectId)}/export/osz?candidate_id=${encodeURIComponent(candidateId)}`;download.setAttribute('aria-disabled','false');}
        download.download=batchDownloadName(item.title||item.name);download.textContent='下载 OSZ';row.append(download);
      }
    }
    list.append(row);
  }
  root.replaceChildren(heading,list);
  updateBatchControls();
}
async function openBatchProject(projectId){
  if(busy)return;
  const request=++projectLoadRequest;
  const project=await api('/api/projects/'+encodeURIComponent(projectId));
  if(request===projectLoadRequest)loadState(project);
}
async function restoreBatchDownloads(batchId){
  const snapshot=await api(`/api/batch/jobs/${encodeURIComponent(batchId)}`);
  if(!snapshot.view_results_available)throw Error('此批次当前没有可恢复的已完成谱包。');
  const items=Array.isArray(snapshot.items)?snapshot.items:[];
  batchPhase='generation';
  batchResultsOnly=true;
  batchAnalysisSnapshot={items:[]};
  batchGenerationAttemptIds=new Set(items.map(batchProjectId).filter(Boolean));
  batchAttemptJobId=batchId;
  batchSnapshot=snapshot;
  setWorkflowMode('batch');
  renderBatchStatus(snapshot);
}
async function monitorBatchGeneration(batch,total){
  if(batchAttemptJobId!==batch.id){
    batchGenerationAttemptIds=new Set((batch.items||[]).map(batchProjectId).filter(Boolean));
    batchAttemptJobId=batch.id;
  }
  batch.generation_total=batch.generation_total||total;
  batchJobId=batch.id;jobId=batch.id;renderBatchStatus(mergeBatchGenerationStatus(batch));
  if(batchCancelRequested){
    try{await api('/api/jobs/'+encodeURIComponent(batchJobId)+'/cancel',{});}
    catch(error){msg('批量生成已提交，但取消请求暂未送达；仍在继续监控任务。',true);}
  }
  let pollFailures=0;
  while(true){
    let current;
    try{current=await api('/api/jobs/'+encodeURIComponent(batchJobId));}
    catch(error){
      pollFailures++;
      const permanent=Number.isInteger(error.status)&&error.status<500&&![408,429].includes(error.status);
      if(permanent||pollFailures>=6){
        const message=permanent
          ?'批量生成状态请求被拒绝（HTTP '+error.status+'），已停止监控。已生成的候选仍保留，可逐首打开。'
          :'连接连续失败，已停止监控批量生成。已生成的候选仍保留，可逐首打开。';
        renderBatchStatus({status:'error',message,items:batchSnapshot?.items||[]});msg(message,true);return 'error';
      }
      $('jobText').textContent='连接暂时中断，正在恢复批量生成状态…';
      msg('批量生成已提交；进度暂时无法读取，正在重试。你仍可点击“取消”。',true);
      await new Promise(resolve=>setTimeout(resolve,1500));
      continue;
    }
    pollFailures=0;$('progress').value=current.progress||0;
    $('jobText').textContent=current.message||'正在统一生成谱面…';
    renderBatchStatus(mergeBatchGenerationStatus(current));
    if(['done','error','cancelled'].includes(current.status)){
      const attempted=batchGenerationRows(current.items||[]);
      const succeeded=attempted.filter(item=>item.status==='done').length;
      const failed=attempted.some(item=>item.status==='error');
      msg(`统一生成结束：${succeeded}/${total} 首成功。`+(failed?'失败的歌曲仍保留，可逐首打开或调整后重试。':''),
        current.status==='error');
      return current.status;
    }
    await new Promise(resolve=>setTimeout(resolve,650));
  }
}
async function runBatchAnalysis(files){
  if(busy)return;
  batchResultsOnly=false;
  if(files.length<2||files.length>20)throw Error('批量分析请一次选择 2–20 首音频。');
  if(files.some(file=>! /\.(mp3|wav|flac|ogg|m4a|opus|aac)$/i.test(file.name)))throw Error('批量分析仅支持 MP3、WAV、FLAC、OGG、M4A、OPUS、AAC 音频。');
  batchImported=[];batchSnapshot=null;batchAnalysisSnapshot=null;batchPhase='analysis';batchImporting=true;batchCancelRequested=false;
  $('batchGeneratedDownloads').replaceChildren();
  $('batchTimingConfirmed').checked=false;
  setBusy(true);$('jobText').textContent='正在逐首导入歌曲…';$('progress').value=0;
  const importedItems=[];
  const cancelImport=()=>{
    renderBatchStatus({status:'cancelled',message:'批量导入已取消；已导入项目保留在本机。',items:importedItems});
    msg('批量导入已取消。已导入的歌曲项目会保留，可稍后单独分析。');
    return 'cancelled';
  };
  try{
    for(let index=0;index<files.length;index++){
      if(batchCancelRequested)return cancelImport();
      const file=files[index];
      $('jobText').textContent='正在导入第 '+(index+1)+'/'+files.length+' 首：'+file.name;
      renderBatchStatus({status:'importing',message:'导入 '+(index+1)+'/'+files.length+'：'+file.name,items:importedItems});
      const data=new FormData();data.append('files',file);
      const project=await api('/api/import',data,true);
      batchImported.push(project);
      importedItems.push({project:project.id,title:project.title||file.name,status:'imported',message:'已创建本地项目，等待分析。'});
      renderBatchStatus({status:'importing',message:'已导入 '+(index+1)+'/'+files.length+' 首。',items:importedItems});
      if(batchCancelRequested)return cancelImport();
    }
    if(batchCancelRequested)return cancelImport();
    $('jobText').textContent='开始并行分析 '+batchImported.length+' 首歌曲…';
    const batch=await api('/api/batch/analyze',{projects:batchImported.map(project=>project.id),...analysisRequestOptions({batch:true}),max_parallel:2});
    batchJobId=batch.id;jobId=batch.id;batchImporting=false;
    renderBatchStatus(batch);
    if(batchCancelRequested){
      try{await api('/api/jobs/'+encodeURIComponent(batchJobId)+'/cancel',{});}
      catch(error){msg('批量任务已提交，但取消请求暂未送达；仍在继续监控任务。',true);}
    }
    let pollFailures=0;
    while(true){
      let current;
      try{current=await api('/api/jobs/'+encodeURIComponent(batchJobId));}
      catch(error){
        pollFailures++;
        const permanent=Number.isInteger(error.status)&&error.status<500&&![408,429].includes(error.status);
        if(permanent||pollFailures>=6){
          const message=permanent
            ?'批量任务状态请求被拒绝（HTTP '+error.status+'），已停止监控。已导入的项目仍保留，可在最近项目中单独打开；任务是否仍在后台运行暂时无法确认。'
            :'连接连续失败，已停止监控批量任务。已导入的项目仍保留，可在最近项目中单独打开；任务是否仍在后台运行暂时无法确认。';
          renderBatchStatus({status:'error',message,items:batchSnapshot?.items||[]});msg(message,true);return 'error';
        }
        $('jobText').textContent='连接暂时中断，正在恢复批量任务状态…';
        msg('批量任务已提交；进度暂时无法读取，正在重试。你仍可点击“取消”。',true);
        await new Promise(resolve=>setTimeout(resolve,1500));
        continue;
      }
      pollFailures=0;
      $('progress').value=current.progress||0;$('jobText').textContent=current.message||'正在批量分析…';
      renderBatchStatus(current);
      if(['done','error','cancelled'].includes(current.status)){
        batchAnalysisSnapshot=current;
        const succeeded=(current.items||[]).filter(item=>item.status==='done').length;
        const hasFailed=(current.items||[]).some(item=>item.status==='error');
        let summary='批量分析结束：'+succeeded+'/'+batchImported.length+' 首成功。'+(hasFailed?'失败的歌曲仍可在下方单独打开。':'');
        try{await refreshRecent();}catch(error){summary+=' 最近项目列表暂未刷新，可稍后重新打开。';}
        msg(summary,current.status==='error');return current.status;
      }
      await new Promise(resolve=>setTimeout(resolve,650));
    }
  }catch(error){
    if(importedItems.length)renderBatchStatus({status:'error',message:'批量任务未能完整启动：'+error.message,items:importedItems});
    throw error;
  }finally{
    batchImporting=false;batchCancelRequested=false;batchJobId=null;jobId=null;setBusy(false);$('batchFiles').value='';updateBatchAvailability();
  }
}
async function runBatchGeneration(){
  if(busy)return;
  batchResultsOnly=false;
  const ready=batchReadyItems();
  if(!ready.length)throw Error('请先完成至少一首歌曲的批量分析。');
  const keys=Number($('batchKeys').value),key_layout=$('batchKeyLayout').value;
  if(!KEY_LAYOUT_CHOICES[keys]?.some(([value])=>value===key_layout))throw Error('请选择与键数匹配的键型预设。');
  const engine=$('batchEngine').value;
  const payload={projects:ready.map(item=>item.project||item.project_id),preset:$('batchPreset').value,
    engine,model_variant:engine==='traditional'?undefined:engine,keys,key_layout,
    density:Number($('batchDensity').value),torosu_density:$('batchTorosuDensity').value,
    torosu_ln_mode:$('batchTorosuLnMode').value,
    difficulty:Number($('batchDifficulty').value),hold_note_ratio:Number($('batchHoldRatio').value),
    lora_path:$('batchLora').value||$('batchLoraPath').value.trim()||null,
    scribe:$('batchScribe').value.trim(),version_suffix:$('batchVersionSuffix').value.trim(),
    timing_confirmed:$('batchTimingConfirmed').checked};
  readyBatchProjectIds=new Set(payload.projects);
  batchGenerationAttemptIds=new Set(payload.projects);
  batchAttemptJobId=null;
  if(!GENERATION_PRESETS[payload.preset]||payload.preset==='torosu_remap')throw Error('请选择有效的音频风格预设。');
  if(engine!=='traditional'&&!v32GenerateAvailable)throw Error('V32 环境尚未就绪，请先查看顶部状态并准备运行环境。');
  batchPhase='analysis';batchSnapshot=batchAnalysisSnapshot;
  $('batchGeneratedDownloads').replaceChildren();$('batchDownloadAll').removeAttribute('href');
  batchGenerationSubmitting=true;batchCancelRequested=false;
  setBusy(true);$('jobText').textContent='正在检查歌曲拍线…';$('progress').value=0;
  try{
    const unstable=await batchUnstableTimingTitles(ready);
    if(batchCancelRequested){msg('批量生成已取消，分析结果保留。');return 'cancelled';}
    if(unstable.length&&!payload.timing_confirmed)
      throw Error(`以下歌曲的自动 Timing 不稳定：${unstable.join('、')}。检查并试听后，或勾选“对拍线不稳定的歌曲也使用自动估计继续生成”再统一生成。`);
    batchPhase='generation';batchCancelRequested=false;
    batchSnapshot=mergeBatchGenerationStatus({status:'queued',batch:true,progress:0,generation_total:ready.length,message:'统一预设已提交，等待生成…',
      items:ready.map(item=>({project:item.project||item.project_id,title:item.title,status:'queued',progress:0,message:'等待生成'}))});
    $('jobText').textContent='正在按统一设置生成 '+ready.length+' 首歌曲…';$('progress').value=0;
    renderBatchStatus(batchSnapshot);
    const batch=await api('/api/batch/generate',payload);
    batchGenerationSubmitting=false;
    return await monitorBatchGeneration(batch,ready.length);
  }catch(error){
    batchPhase='analysis';batchSnapshot=batchAnalysisSnapshot;
    if(batchSnapshot)renderBatchStatus(batchSnapshot);
    throw error;
  }finally{batchGenerationSubmitting=false;batchJobId=null;jobId=null;batchCancelRequested=false;setBusy(false);}
}
async function importFiles(files){if(!files.length||busy)return;setBusy(true);$('jobText').textContent='导入素材…';try{const data=new FormData();for(const f of files)data.append('files',f);loadState(await api('/api/import',data,true),{newAudioProject:true});msg(state.mode==='remap'?'原谱已导入，可直接严格重新定轨。':'音频已导入，点击分析音频开始。');}finally{setBusy(false);$('files').value='';}}
async function generate(region){
  $('offset').value=chartOffset(Number($('offset').value));
  const preset=$('preset').value,engine=$('singleEngine').value;
  const currentSettings=settings();
  if(region&&engine!=='traditional')
    throw Error('V32 当前只支持整谱生成；局部重写请切换为传统 CPU 引擎，以保留区域外已有物件。');
  if(region&&currentSettings.layout_engine==='torosu')
    throw Error('Torosu 不支持局部重写；请在高级设置中将定轨引擎切换为 beam 或 classic 后重试。');
  if(engine!=='traditional'&&!region){
    if($('mode').value==='remap')throw Error('V32 音频大脑不用于已有谱严格转键；请切换传统 Torosu。');
    if(preset==='custom')throw Error('V32 生成请选择五大风格预设之一。');
    const values=GENERATION_PRESETS[preset]||GENERATION_PRESETS.hybrid_adaptive;
    return runJob('v32-generate',{
      preset,keys,key_layout:$('keyLayout').value,bpm:Number($('bpm').value),
      offset:Number($('offset').value),division:Number($('division').value),
      audio_shift_ms:clockShift(),difficulty:Number($('v32Difficulty').value),
      model_variant:engine,generation_strategy:$('v32Strategy').value,
      hold_note_ratio:Number($('v32HoldRatio').value),
      mania_column_temperature:Number($('v32Temperature').value),
      torosu_density:$('v32Density').value,
      torosu_keep_jacks:values.torosu_keep_jacks!==false,torosu_ln_mode:$('v32LnMode').value,
      lora_path:$('v32LoraSelect').value||$('v32LoraPath').value.trim()||null,
      metadata:{creator:$('singleScribe').value.trim()||'zhyzx35607',version_suffix:$('singleVersionSuffix').value.trim()},
      timing_confirmed:$('timingConfirmed').checked,
    });
  }
  const payload={...(preset==='custom'?{}:{preset}),settings:settings(),mode:$('mode').value,bpm:Number($('bpm').value),
    offset:Number($('offset').value),division:Number($('division').value),audio_shift_ms:clockShift(),
    snap:true,timing_confirmed:$('timingConfirmed').checked,
    metadata:{creator:$('singleScribe').value.trim()||'zhyzx35607',version_suffix:$('singleVersionSuffix').value.trim()},
    ...(region?{region}:{})};
  return runJob('generate',payload);
}
numbers.forEach(k=>$(k).addEventListener('input',updateOutputs));
$('batchDownloadAll').setAttribute('aria-disabled','true');
numbers.forEach(k=>$(k).addEventListener('input',()=>{syncPresetLabel();updateGridStatus();}));
document.querySelectorAll('[data-key]').forEach(b=>b.onclick=()=>{keys=Number(b.dataset.key);updateKeys();syncPresetLabel();updateGridStatus();window.updateEditorLocks?.();});
$('keyLayout').addEventListener('change',()=>{syncPresetLabel();updateGridStatus();});
$('focus').addEventListener('change',()=>{syncPresetLabel();updateGridStatus();});
$('layoutEngine').addEventListener('change',()=>{syncPresetLabel();updateGridStatus();});
$('referenceStyleFile').onchange=guarded(async e=>{
  const file=e.target.files[0];if(!file)return;if(!state)throw Error('请先导入要制谱的音频或项目');if(busy)return;
  const draft=captureTimingDraft(['analysisEngine']);
  setBusy(true);
  try{
    const data=new FormData();data.append('file',file);data.append('settings',JSON.stringify(settings()));
    loadState(await api(`/api/projects/${state.id}/reference-style`,data,true));
    restoreTimingDraft(draft);
    msg('已套用参考谱的全局偏好，请重新生成候选。已有 API 分段建议仍会应用，可按需清除。');
  }finally{setBusy(false);$('referenceStyleFile').value='';}
});
$('files').onchange=guarded(e=>importFiles(e.target.files));
$('batchFiles').onchange=updateBatchAvailability;
$('batchFiles').disabled=false;
$('batchAnalyze').onclick=guarded(()=>runBatchAnalysis([...$('batchFiles').files]));
$('batchGenerate').onclick=guarded(()=>runBatchGeneration());
$('batchDownloadAll').onclick=()=>msg('已开始下载批量谱包；服务端逐首打包，不会在浏览器中缓存全部音频。');
$('batchKeys').onchange=()=>{updateBatchKeyLayout();updateBatchControls();};
$('batchPreset').onchange=()=>{const presetId=$('batchPreset').value,preset=GENERATION_PRESETS[presetId];if(preset){$('batchDensity').value=preset.density;$('batchTorosuDensity').value=PRESET_TOROSU_DEFAULTS[presetId]||'dense';$('batchTorosuLnMode').value=preset.torosu_ln_mode||'keep';$('batchHoldRatio').value=PRESET_V32_HOLD[presetId]??.3;$('batchHoldRatioOut').textContent=`${Math.round(Number($('batchHoldRatio').value)*100)}%`;}updateBatchControls();};
$('batchDensity').oninput=updateBatchControls;
$('batchTorosuDensity').onchange=updateBatchControls;
$('batchTorosuLnMode').onchange=updateBatchControls;
$('batchHoldRatio').oninput=()=>{$('batchHoldRatioOut').textContent=`${Math.round(Number($('batchHoldRatio').value)*100)}%`;};
$('batchEngine').onchange=()=>{batchEngineTouched=true;if($('batchEngine').value!=='traditional'&&!v32GenerateAvailable)msg('顶部 V32 状态未就绪；准备环境后即可运行 V32 批量。',true);};
$('batchKeyLayout').onchange=updateBatchControls;
updateBatchKeyLayout();updateBatchControls();
$('batchHistory').addEventListener('click',guarded(async event=>{
  const watch=event.target.closest('button[data-watch-batch]');
  if(watch)await reconnectPersistentBatch(watch.dataset.watchBatch);
  const button=event.target.closest('button[data-resume-batch]');
  if(button)await resumePersistentBatch(button.dataset.resumeBatch);
  const results=event.target.closest('button[data-show-batch-results]');
  if(results)await restoreBatchDownloads(results.dataset.showBatchResults);
}));
$('batchStatus').addEventListener('click',guarded(async event=>{
  const button=event.target.closest('button[data-open-project]');
  if(button)await openBatchProject(button.dataset.openProject);
}));
$('batchStatus').addEventListener('click',guarded(async event=>{
  const button=event.target.closest('button[data-resume-batch]');
  if(button)await resumePersistentBatch(button.dataset.resumeBatch);
}));
document.querySelectorAll('[data-work-mode]').forEach(button=>button.onclick=()=>setWorkflowMode(button.dataset.workMode));
setWorkflowMode('single');
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('drag');};$('drop').ondragleave=()=>$('drop').classList.remove('drag');$('drop').ondrop=guarded(e=>{e.preventDefault();$('drop').classList.remove('drag');return importFiles(e.dataTransfer.files);});
$('recent').onchange=guarded(async e=>{const id=e.target.value;if(!id)return;const request=++projectLoadRequest;const project=await api('/api/projects/'+id);if(request===projectLoadRequest)loadState(project);});
async function switchSavedSelection(kind,payload){
  if(busy)return;
  projectLoadRequest++;setBusy(true);$('jobText').textContent='正在切换…';
  try{loadState(await api(`/api/projects/${state.id}/${kind}`,payload));}
  finally{$('charts').value=state.original||'';$('variants').value=state.active;setBusy(false);}
}
$('charts').onchange=guarded(e=>switchSavedSelection('chart',{path:e.target.value}));
$('variants').onchange=guarded(e=>{const index=Number(e.target.value);return switchSavedSelection('variant',{index,candidate_id:state?.variants[index]?.candidate_id});});
$('analyze').onclick=guarded(()=>runJob('analyze',analysisRequestOptions()));$('generate').onclick=guarded(()=>generate());
$('another').onclick=guarded(()=>{$('seed').value=Math.floor(Math.random()*2147483647);return generate();});
$('regenerate').onclick=guarded(()=>generate([Number($('regionStart').value)*1000,Number($('regionEnd').value)*1000]));
$('cancel').onclick=guarded(()=>{if(batchImporting||batchGenerationSubmitting){batchCancelRequested=true;$('jobText').textContent=batchImporting?'正在取消批量导入…':'正在取消批量生成…';return jobId?api('/api/jobs/'+encodeURIComponent(jobId)+'/cancel',{}):null;}return jobId?api('/api/jobs/'+encodeURIComponent(jobId)+'/cancel',{}):null;});
$('mode').onchange=()=>{setMode();cueClock.reset();cancelAuditionPlayback();clearTimingDiagnostics();renderTimingLocations();draw();};
$('plan').onclick=guarded(async()=>{
  if(busy)return;const draft=captureTimingDraft(['analysisEngine']);
  const status=await runJob('plan',{endpoint:$('endpoint').value,model:$('model').value,key:$('apiKey').value,instruction:$('instruction').value,settings:settings()});
  if(status==='done')restoreTimingDraft(draft);
});
$('clearPlan').onclick=guarded(async()=>{
  if(!state||busy)return;const draft=captureTimingDraft(['analysisEngine']);setBusy(true);
  try{loadState(await api(`/api/projects/${state.id}/clear-plan`,{}));restoreTimingDraft(draft);}
  finally{setBusy(false);}
});
$('engineInstall').onclick=guarded(()=>runJob('engine-install',{}));
$('engineGenerate').onclick=guarded(()=>runJob('engine-generate',{settings:settings(),seconds:Number($('engineSeconds').value),use_timing:$('engineUseTiming').checked,bpm:Number($('bpm').value),offset:Number($('offset').value),division:Number($('division').value),audio_shift_ms:clockShift(),timing_confirmed:$('timingConfirmed').checked}));
$('v32Install').onclick=guarded(()=>runGlobalV32Install());
$('headerV32Install').onclick=guarded(()=>runGlobalV32Install());
$('v32HoldRatio').oninput=updateV32Outputs;$('v32Temperature').oninput=updateV32Outputs;
$('v32Generate').onclick=guarded(()=>{$('singleEngine').value=$('v32Model').value;return generate();});
$('preset').onchange=()=>applyGenerationPreset($('preset').value);
$('singleEngine').onchange=()=>{singleEngineTouched=true;updateSingleEngineControls();$('v32PanelControls').classList.toggle('hidden',$('singleEngine').value==='traditional');};
$('v32LoraSelect').onchange=()=>{if($('v32LoraSelect').value)$('v32LoraPath').value='';};
$('batchLora').onchange=()=>{if($('batchLora').value)$('batchLoraPath').value='';};
$('torosuKeepJacks').onchange=()=>{syncPresetLabel();updateGridStatus();};
$('torosuLegacyDensity').onchange=()=>{syncPresetLabel();updateGridStatus();};
$('torosuLnMode').onchange=()=>{syncPresetLabel();updateGridStatus();};
$('singleEngine').value='v32-mini';updateSingleEngineControls();
if(window.ManiaLabChartPreview?.mount)previewController=window.ManiaLabChartPreview.mount($('chartPreviewHost'),audio);
function setView(v){const changed=view!==v;cueClock.reset();cancelAuditionPlayback();view=v;selected=null;updateVoiceOptions();setPlaybackSource();updateAudioStatus();if(changed)window.resetWaveDetail?.();window.renderNoteEditor?.();$('chartView').classList.toggle('active',v==='chart');$('evidenceView').classList.toggle('active',v==='evidence');draw();}
function chooseView(v){setView(v);if($('waveDetailPanel').open&&state?.analysis)guarded(()=>loadWaveDetail())();}
$('chartView').onclick=()=>chooseView('chart');$('evidenceView').onclick=()=>chooseView('evidence');$('window').onchange=draw;
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
const importedTimingCache=new WeakMap();
function audioMeter(bpm,offset){
  const saved=variant(),shift=clockShift();
  if(saved?.mode==='audio'&&bpm===saved.bpm&&offset===saved.offset&&shift===(saved.audio_shift_ms??0)){
    return saved.meter??4;
  }
  const detected=state?.analysis?.tempo;
  if(detected&&bpm===detected.bpm&&offset===autoChartOffset(detected.offset-shift))return state.detected_meter??4;
  return 4;
}
function timingPoints(){
  if($('mode').value==='remap'){
    const lines=state?.input_timing;
    if(!Array.isArray(lines))return [];
    if(!importedTimingCache.has(lines)){
      const redLines=new Map();
      for(const line of lines){
        if(typeof line!=='string')continue;
        const fields=line.split(',').map(field=>field.trim());
        if(!fields[0]||!fields[1]||(fields.length>=7&&fields[6]!=='1'))continue;
        const offset=Number(fields[0]),beat=Number(fields[1]);
        if(!Number.isFinite(offset)||!Number.isFinite(beat)||beat<=0)continue;
        const meter=/^[+]?\d+$/.test(fields[2]||'')?Number(fields[2]):4;
        redLines.set(offset,{offset,beat,meter:Number.isSafeInteger(meter)&&meter>0?meter:4});
      }
      importedTimingCache.set(lines,[...redLines.values()].sort((a,b)=>a.offset-b.offset));
    }
    return importedTimingCache.get(lines);
  }
  const bpm=Number($('bpm').value),offset=chartOffset(Number($('offset').value));
  return Number.isFinite(bpm)&&bpm>=10&&bpm<=1000&&Number.isFinite(offset)?[{offset,beat:60000/bpm,meter:audioMeter(bpm,offset)}]:[];
}
function beatsBetween(start,end){
  const points=timingPoints(),out=[];
  for(const [i,p] of points.entries()){
    const begin=i===0?start*1000:Math.max(start*1000,p.offset);
    const finish=Math.min(end*1000,points[i+1]?.offset??Infinity);
    if(begin>=finish)continue;
    const first=Math.ceil((begin-p.offset)/p.beat),stop=Math.ceil((finish-p.offset)/p.beat);
    if(!Number.isSafeInteger(first)||!Number.isSafeInteger(stop)||stop-first>10000-out.length)return [];
    for(let j=first;j<stop;j++){
      const time=p.offset+j*p.beat;
      if(time>=begin&&time<finish)out.push({time:time/1000,accent:j%p.meter===0});
    }
  }
  return out;
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
function timingChanged(preserveClockAnchor=false){ if(!preserveClockAnchor)clockAnchor=Number($('offset').value)+clockShift();cueClock.reset();$('timingConfirmed').checked=false;clearTimingDiagnostics();renderTimingLocations();lastScheduledBeat=-Infinity;updateGridStatus();draw();window.drawWaveDetail?.(); }
['bpm','offset','division'].forEach(id=>$(id).addEventListener('input',()=>timingChanged(id!=='offset')));
$('audioShift').addEventListener('change',()=>{const value=clockShift();if(!Number.isFinite(value)||Math.abs(value)>100)return;$('offset').value=autoChartOffset(clockAnchor-value);lastClockShift=value;tempoChoices=tempoChoices.map(c=>({...c,offset:autoChartOffset(c.audio_offset-value)}));showTempoChoices();timingChanged(true);msg(`制谱提前量 ${value} ms；Offset 已同步调整。原始采音保留，需重新生成整谱。`);});
$('tempoCandidates').onchange=()=>{const i=$('tempoCandidates').value;if(i==='')return;const c=tempoChoices[Number(i)];$('bpm').value=c.bpm;$('offset').value=c.offset;$('division').value=c.division;clockAnchor=c.audio_offset;timingChanged(true);if(c.precision_suggestion)msg(`已采用 ${c.bpm} BPM 精确候选；请在歌曲开头和结尾试听，确认没有逐渐漂移。`);};
$('halfBpm').onclick=()=>{$('bpm').value=Math.max(10,Number($('bpm').value)/2);timingChanged(true);};
$('doubleBpm').onclick=()=>{$('bpm').value=Math.min(1000,Number($('bpm').value)*2);timingChanged(true);};
$('applyTiming').onclick=guarded(()=>{
  const text=$('referenceTiming').value;
  const bpm=text.match(/BPM\s*[:：=]\s*([\d.]+)/i),offset=text.match(/Offset\s*[:：=]\s*([+-]?[\d.]+)/i);
  const pair=text.trim().split(/[\s,，;；]+/).map(Number);
  const b=bpm?Number(bpm[1]):pair.length===2?pair[0]:NaN,o=offset?Number(offset[1]):pair.length===2?pair[1]:NaN;
  if(!Number.isFinite(b)||b<10||b>1000||!Number.isFinite(o)||Math.abs(o)>600000)throw Error('请粘贴 BPM: 210 与 Offset: 91，或两个用空格分隔的数字');
  const roundedOffset=chartOffset(o);
  $('bpm').value=b;$('offset').value=roundedOffset;timingChanged();msg(`已应用参考：${b} BPM，Offset ${roundedOffset} ms。请试听确认后重新生成整谱。`);
});
$('refitTiming').onclick=guarded(async()=>{
  if(!state)throw Error('请先分析音频');
  normalizeTimingOffset();
  const requestNumber=++timingRequest,pid=state.id,bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value),shift=clockShift();
  const r=await api(`/api/projects/${pid}/timing-refit`,{bpm,offset,division,audio_shift_ms:shift});
  if(state.id!==pid||timingRequest!==requestNumber||Number($('bpm').value)!==bpm||Number($('offset').value)!==offset||Number($('division').value)!==division||clockShift()!==shift)return;
  $('offset').value=r.offset;timingChanged();showTimingDiagnostics(r,`按 ${r.bpm} BPM 重新估计 Offset：${r.offset} ms。`);
});
$('checkTiming').onclick=guarded(async()=>{
  if(!state)throw Error('请先分析音频');
  normalizeTimingOffset();
  const requestNumber=++timingRequest,pid=state.id,bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value),shift=clockShift();
  const r=await api(`/api/projects/${pid}/timing-check`,{bpm,offset,division,audio_shift_ms:shift});
  if(state.id!==pid||timingRequest!==requestNumber||Number($('bpm').value)!==bpm||Number($('offset').value)!==offset||Number($('division').value)!==division||clockShift()!==shift)return;
  showTimingDiagnostics(r);
});
['metronome','clicks','releases'].forEach(id=>$(id).onchange=guarded(async()=>{cueClock.reset();if($(id).checked)await cueClock.enable();}));
$('voice').onchange=()=>{if(!state)return;cueClock.reset();cancelAuditionPlayback();setPlaybackSource();updateAudioStatus();draw();};
$('auditionHere').onclick=()=>{$('auditionStart').value=currentTime().toFixed(1);};
$('auditionLoop').onchange=()=>{$('auditionAudio').loop=$('auditionLoop').checked;};
$('auditionApply').onclick=()=>{$('offset').value=chartOffset(Number($('auditionOffset').value));timingChanged();msg('已采用对照 B 的 Offset；请检查整曲后重新生成。');};
async function audition(mode,compare=false){
  const originalBeats=mode==='beats'&&$('mode').value==='remap';
  const remapMix=originalBeats||(mode==='chart'&&variant()?.mode==='remap');
  if(!state?.analysis&&!(remapMix&&state?.audio))throw Error('请先分析音频');
  if((mode==='chart'||mode==='evidence')&&view!==mode)chooseView(mode);
  cancelAuditionPlayback();const requestNumber=++auditionRequest;
  const pid=state.id,offset=chartOffset(Number($(compare?'auditionOffset':'offset').value));
  const payload={mode,context:view,audio:state.audio||'',source_identity:$('voice').value==='mix'?null:playbackSourceIdentity(),
    start_ms:Number($('auditionStart').value)*1000,duration_ms:Number($('auditionLength').value)*1000,
    bpm:Number($('bpm').value),offset,division:Number($('division').value),audio_shift_ms:clockShift(),releases:true,voice:$('voice').value};
  if(mode==='chart')payload.candidate_id=variant()?.candidate_id??null;
  if(mode==='beats')payload.beat_clock=originalBeats?'original':'fixed';
  $('auditionNote').textContent='正在混合音乐和提示音…';
  const response=await fetch(`/api/projects/${pid}/audition`,{method:'POST',headers:{'X-ManiaLab-Token':token,'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!response.ok)throw Error((await response.json()).error||'试听失败');
  const blob=await response.blob();if(state.id!==pid||requestNumber!==auditionRequest)return;
  audio.pause();$('auditionAudio').pause();if(auditionUrl)URL.revokeObjectURL(auditionUrl);
  auditionUrl=URL.createObjectURL(blob);$('auditionAudio').src=auditionUrl;$('auditionAudio').loop=$('auditionLoop').checked;
  $('auditionNote').textContent=payload.beat_clock==='original'?`原谱拍线 · ${payload.start_ms/1000} 秒开始，重音按各段拍号播放。`:
    mode==='beats'?`${compare?'B':'A'}：${payload.bpm} BPM / ${offset} ms，每小节 ${response.headers.get('X-ManiaLab-Audition-Meter')} 拍，${payload.start_ms/1000} 秒开始。`:
    `${mode==='chart'?'候选谱面':'原始采音'}：${payload.start_ms/1000} 秒开始，高音提示起点、低音提示持续音结束。`;
  await $('auditionAudio').play();
}
$('auditionA').onclick=guarded(()=>audition('beats'));$('auditionB').onclick=guarded(()=>audition('beats',true));
$('auditionChart').onclick=guarded(()=>audition('chart'));$('auditionEvidence').onclick=guarded(()=>audition('evidence'));
audio.addEventListener('play',()=>{$('auditionAudio').pause();});
updateOutputs();updateV32Outputs();window.addEventListener('DOMContentLoaded',guarded(refreshRecent),{once:true});draw();requestAnimationFrame(tick);
