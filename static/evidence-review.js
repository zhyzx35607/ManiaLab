'use strict';

const reviewVoiceNames={mix:'混音 / 原谱',drums:'鼓 / 打击',bass:'贝斯',vocals:'人声',other:'其余伴奏',guitar:'吉他',piano:'钢琴'};
const signedMilliseconds=value=>`${value>0?'+':''}${Number(value.toFixed(3))} ms`;
const validEvidenceInterval=value=>value&&Number.isFinite(value.start)&&Number.isFinite(value.end)&&value.start>=0&&value.end>=value.start;
const noteWasEdited=(note,candidate=variant())=>(candidate?.edits||[]).some(edit=>edit.op==='update'&&edit.after?.id===note.id);

function savedNoteEvidence(note,candidate=variant()){
  // IDs can be reused by a later analysis. Only the selected candidate's own
  // snapshot or legacy decision trace can establish its original endpoints.
  const snapshot=candidate?.source_evidence?.[note.id];
  // A new-format regional result may lock a legacy note with no old snapshot.
  // Its missing entry means unknown; this generation's decisions are unrelated.
  if(candidate&&Object.prototype.hasOwnProperty.call(candidate,'source_evidence'))
    return {detected:validEvidenceInterval(snapshot?.detected)?snapshot.detected:null,
      expressed:validEvidenceInterval(snapshot?.expressed)?snapshot.expressed:null};
  // Legacy regional traces describe the new analysis, not locked old objects.
  // Match generation.py: chart starts in [a,b) are regenerated; others lock.
  const region=candidate?.region;
  if(region!=null&&(!Array.isArray(region)||region.length!==2||!region.every(Number.isFinite)||
    region[0]<0||region[0]>=region[1]||!(region[0]<=note.start&&note.start<region[1])))return {};
  const matches=(candidate?.selection?.decisions||[]).filter(row=>row.id===note.id&&row.decision==='selected');
  if(matches.length!==1||!validEvidenceInterval(matches[0]))return {};
  const row=matches[0];
  return {detected:{start:row.start,end:row.end},
    expressed:Number.isFinite(row.expression_end)&&row.expression_end>=row.start?{start:row.start,end:row.expression_end}:null};
}

function releaseDescription(reason){
  const known={
    model_predicted_offset:'模型预测音尾，需回听核对',
    release_regression:'模型释放回归估计，需回听核对',
    frame_decay:'模型持续激活消退，需回听核对',
    vocal_activity_decay:'人声活动消退形成的乐句尾，需回听核对',
    same_pitch_reattack:'下一次同音高起音形成的音乐边界，非独立放音观测',
    reattack:'同音高重新起音截断，未观测到音尾',
    frame_inactive:'持续激活未成立，未观测到音尾',
    duration_limit:'达到模型时长上限，未观测到音尾',
    input_boundary:'到达输入边界，未观测到音尾',
    forced_segment_boundary:'分段强制截断，未观测到音尾',
    unresolved_or_postprocessed:'音尾未决或经过后处理，未观测到音尾',
    explicit_model_note_off:'模型明确输出结束事件，需回听核对',
  };
  return reason?(known[reason]||`未知来源类型（${reason}）`):'未知（未记录来源）';
}

function inspectNote(note){
  selected=note.id;window.renderNoteEditor?.(note);
  const candidate=variant(),raw=view==='evidence',manual=note.source==='manual',edited=!raw&&noteWasEdited(note,candidate);
  const origin=manual?'人工添加物件':note.source==='import'?'原谱物件':note.source==='pitch'?`模型音高 MIDI ${note.pitch} · 候选权重 ${note.strength}`:'瞬态候选';
  const parts=[`${note.end>note.start?'LN / 持续音':'短键 / 瞬态'} · 起点 ${note.start} ms · 终点 ${note.end} ms · 长度 ${note.end-note.start} ms`,origin];
  if(edited)parts.push('人工修改（当前起止或轨道包含编辑）');
  if(note.lane>=0)parts.push(`轨道 ${note.lane+1}`);
  parts.push(`声部 ${reviewVoiceNames[note.voice]||note.voice||'mix'}`);
  if(manual){
    parts.push('人工设定起止并按拍线量化；没有采音证据');
  }else if(note.source==='import'||(!raw&&candidate?.mode==='remap')){
    parts.push('起止来自原谱，不代表音频检测结果');
  }else{
    const evidence=raw?{detected:note,expressed:null}:savedNoteEvidence(note,candidate);
    if(evidence.detected){
      parts.push(`原始候选起点（原音频）${evidence.detected.start} ms`, `原始候选尾（原音频）${evidence.detected.end} ms`);
    }else parts.push('原始候选尾：未知（此候选未保存可核对的原始时间）');
    if(!raw){
      if(evidence.expressed)parts.push(`量化前表达（原音频）${evidence.expressed.start}–${evidence.expressed.end} ms`);
      const movement=edited?'人工修改后差值':'量化移动';
      if(Number.isFinite(note.evidence_start))parts.push(`量化前起点（制谱时间）${note.evidence_start} ms（${movement} ${signedMilliseconds(note.start-note.evidence_start)}）`);
      if(Number.isFinite(note.evidence_end))parts.push(`量化前表达尾（制谱时间）${note.evidence_end} ms（${movement} ${signedMilliseconds(note.end-note.evidence_end)}）`);
      else parts.push('量化前表达尾：未知');
    }
    if(note.phrase){
      parts.push(`跨音高持续段 · 支撑候选 ${(note.support||[]).length} 个 · 活动覆盖 ${((note.support_coverage||0)*100).toFixed(0)}%`);
    }
    if(note.source==='pitch'||note.release_reason){
      parts.push('音尾来源：'+releaseDescription(raw?note.release_reason:evidence.expressed?.release_reason??note.release_reason));
      if(!raw&&evidence.detected?.release_reason&&evidence.detected.release_reason!==evidence.expressed?.release_reason&&evidence.expressed?.release_reason)
        parts.push('原始候选尾来源：'+releaseDescription(evidence.detected.release_reason));
      if(edited)parts.push('来源描述保留原证据，不为人工修改后的音尾背书');
    }
  }
  $('inspector').textContent=parts.join(' · ');draw();
}

async function locateReviewNote(note,endpoint){
  const candidate=variant(),shift=candidate?.mode==='audio'?(candidate.audio_shift_ms||0):0;
  const audioTime=note[endpoint]+shift;
  setView('chart');
  const voice=$('voice'),wanted=!voice.disabled&&[...voice.options].some(option=>option.value===note.voice)?note.voice:'mix';
  if(voice.value!==wanted){voice.value=wanted;voice.dispatchEvent(new Event('change'));}
  audio.pause();seek(Math.max(0,audioTime/1000-.3));inspectNote(note);
  if(window.loadWaveDetail)await loadWaveDetail(waveChartTime(note[endpoint],shift,waveShift()));
}

function renderReview(){
  const candidate=variant(),body=$('reviewVoices'),events=$('reviewEvents');body.replaceChildren();events.replaceChildren();
  if(!candidate){$('reviewSummary').textContent='生成后显示声部取舍、长条和需要回听的位置。';return;}
  const notes=candidate.notes;
  for(const voice of [...new Set(notes.map(note=>note.voice||'mix'))]){
    const group=notes.filter(note=>(note.voice||'mix')===voice),holds=group.filter(note=>note.end>note.start),row=document.createElement('tr');
    for(const value of [reviewVoiceNames[voice]||voice,group.length,holds.length,Math.max(0,...holds.map(note=>note.end-note.start))+' ms']){
      const cell=document.createElement('td');cell.textContent=value;row.append(cell);
    }body.append(row);
  }
  const displaced=(endpoint,evidence)=>notes.filter(note=>note.source!=='manual'&&Number.isFinite(note[evidence])&&Math.abs(note[endpoint]-note[evidence])>30)
    .sort((a,b)=>Math.abs(b[endpoint]-b[evidence])-Math.abs(a[endpoint]-a[evidence]));
  const heads=displaced('start','evidence_start'),tails=displaced('end','evidence_end').filter(note=>note.end>note.start);
  const phrase=notes.filter(note=>note.phrase&&note.end>note.start).length;
  $('reviewSummary').textContent=candidate.mode==='remap'?'严格保时：原谱起止时间和 LN 类型保留。手感与声部表达仍需试听。':
    `保留 ${phrase} 段跨音高人声长条；${heads.length} 个起音、${tails.length} 个 LN 尾端与量化前表达相差超过 30 ms（含人工修改，人工添加不计入）。分别列出偏移最大的前 12 处，点击定位对应声部；这些是回听线索，不是质量评分。`;
  if(candidate.mode==='audio')for(const [kind,label,endpoint,evidence,items] of [
    ['head','起音','start','evidence_start',heads],['tail','LN 尾端','end','evidence_end',tails]]){
    const group=document.createElement('div'),caption=document.createElement('p');caption.className='muted compact';caption.textContent=`${label}偏移 · ${items.length} 处`;group.append(caption);
    for(const note of items.slice(0,12)){
      const button=document.createElement('button');button.className='quiet';button.dataset.reviewKind=kind;
      button.textContent=`${label} ${fmt(note[endpoint]/1000)} · ${reviewVoiceNames[note.voice]||note.voice} · ${signedMilliseconds(note[endpoint]-note[evidence])}`;
      button.onclick=guarded(()=>locateReviewNote(note,endpoint));group.append(button);
    }events.append(group);
  }
  const review=candidate.report?.review;
  if(review){
    const counts=review.counts||{};$('reviewSummary').textContent+=` 手感检查：${counts.short_release||0} 次短放键接续、${counts.fast_repeat||0} 次快速同轨、${counts.held_hand_load||0} 处同手 LN 配合密集。它们可能是有意设计，请结合目标难度试打。`;
    for(const item of (review.locations||[]).slice(0,16)){
      const button=document.createElement('button');button.className='quiet';button.textContent=`${fmt(item.start_ms/1000)} · ${item.text}`;
      button.onclick=()=>{setView('chart');seek(Math.max(0,item.start_ms/1000-.4));const note=notes.find(note=>note.start===item.start_ms&&(item.lane===null||note.lane===item.lane));if(note)inspectNote(note);};events.append(button);
    }
  }
}
