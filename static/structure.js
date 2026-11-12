'use strict';
const structurePanel=document.createElement('details');
structurePanel.id='structurePanel';structurePanel.className='review-panel';structurePanel.open=true;
const structureTitle=document.createElement('summary');structureTitle.textContent='先规划音乐 · 段落与声部表达';
const structureInfo=document.createElement('p');structureInfo.className='muted compact';
const structureActions=document.createElement('div');structureActions.className='structure-actions';
const sectionDetect=document.createElement('button');sectionDetect.id='sectionDetect';sectionDetect.textContent='检测音乐段落';sectionDetect.className='secondary';
const bookmarkImport=document.createElement('button');bookmarkImport.id='bookmarkImport';bookmarkImport.textContent='采用原谱书签';bookmarkImport.className='quiet';
const kiaiImport=document.createElement('button');kiaiImport.id='kiaiImport';kiaiImport.textContent='采用原谱 Kiai 切换';kiaiImport.className='quiet';
const structureBoundaries=document.createElement('input');structureBoundaries.id='structureBoundaries';structureBoundaries.placeholder='分界秒数，以逗号分隔，例如 14.273, 42.508';
const boundaryLabel=document.createElement('label');boundaryLabel.className='field';boundaryLabel.textContent='编辑分界 / 原始音频秒数';boundaryLabel.append(structureBoundaries);
const sectionSave=document.createElement('button');sectionSave.id='sectionSave';sectionSave.textContent='保存分界';sectionSave.className='quiet';
const expressionAuto=document.createElement('button');expressionAuto.id='expressionAuto';expressionAuto.textContent='重新建议声部方案';expressionAuto.className='quiet';
const expressionSave=document.createElement('button');expressionSave.id='expressionSave';expressionSave.textContent='保存声部方案';expressionSave.className='secondary';
const structureRows=document.createElement('div');structureRows.id='structureRows';
const expressionAudit=document.createElement('div');expressionAudit.id='expressionAudit';
structureActions.append(sectionDetect,bookmarkImport,kiaiImport,sectionSave,expressionAuto,expressionSave);
structurePanel.append(structureTitle,structureInfo,boundaryLabel,structureActions,structureRows,expressionAudit);
document.querySelector('article .project-head').after(structurePanel);
let expressionDraft=null;
const sourceNames={mix:'混音（未分轨）',vocals:'人声',drums:'鼓 / 打击',bass:'贝斯',other:'其余伴奏',piano:'钢琴',guitar:'吉他'};
const omissionNames={weak_detection:'检测较弱',voice_not_selected:'本段未采或声部静音',represented_by_phrase:'已由连续乐句表示',represented_by_chord:'该声部在同拍位置已用较少层数表示',phrase_not_selected:'所属连续乐句未被保留',group_capacity_or_density:'完整音组超出密度或占轨预算',duplicate_pitch:'同声部重复候选'};

function expressionOptions(voice){
  const items=[['sustain','持续音用 LN，短音用短键'],['articulation','连续起音优先，组尾可延长'],['onsets','只表示起音'],['omit','本段不采']];
  if(voice==='vocals')items.splice(1,0,['phrase','连续人声乐句（需回听）']);
  return items;
}
function instrumentName(id){
  const tsumugi=/^tsumugi:[a-z0-9_]+:[a-f0-9]{16}:class:(\d+):([a-z0-9_]+)$/.exec(id);
  if(tsumugi){
    const names={accordion_family:'手风琴类',acoustic_bass:'原声低音提琴',acoustic_guitar:'原声吉他',brass:'铜管',choir:'合唱',
      chromatic_percussion:'有音高打击乐',distorted_guitar:'失真吉他',drums:'鼓',electric_bass:'电贝斯',
      electric_guitar_clean:'清音电吉他',electric_guitar_muted:'闷音电吉他',electric_piano:'电钢琴',ethnic:'民族乐器',
      flute_pipe:'笛类',guitar_harmonics:'吉他泛音',harmonica:'口琴',orchestra_hit:'管弦齐奏重音',orchestral_harp:'竖琴',
      orchestral_woodwind:'管弦木管',organ:'风琴',percussive_fx:'打击音效',piano:'钢琴',pizzicato_strings:'拨奏弦乐',
      plucked_keyboard:'拨弦键盘',sax:'萨克斯',slap_bass:'击弦贝斯',sound_fx:'音效',strings:'弦乐',synth_bass:'合成贝斯',
      synth_fx:'合成音效',synth_lead:'合成器主音',synth_pad:'合成器铺底',timpani:'定音鼓',melody:'旋律声部',
      vocal_harmony:'人声和声',wind_chimes:'风铃'};
    return `${names[tsumugi[2]]||tsumugi[2].replaceAll('_',' ')} · 模型类别`;
  }
  const match=/^yourmt3:program:(\d+)$/.exec(id);
  if(!match)return `模型类别 ${id}`;
  const program=Number(match[1]);
  const specific={0:'三角钢琴',2:'电声三角钢琴',8:'钢片琴',16:'拉杆风琴',40:'小提琴',41:'中提琴',42:'大提琴',43:'低音提琴',44:'颤音弦乐',45:'拨奏弦乐',46:'竖琴',47:'定音鼓',48:'弦乐合奏',49:'弦乐合奏',100:'主唱',101:'合唱'};
  const families=['钢琴','有音高打击乐','风琴','吉他','贝斯','弦乐','弦乐合奏／合唱','铜管','簧管','笛类','合成器主音','合成器铺底'];
  return `${specific[program]||families[Math.floor(program/8)]||'未知音色'} · 类别 ${program}`;
}

function preserveTimingLoad(next){
  const retained=Object.fromEntries(['bpm','offset','division','audioShift','auditionOffset','mode'].map(k=>[k,$(k).value]));
  const confirmed=$('timingConfirmed').checked,anchor=clockAnchor;
  loadState(next);for(const [k,v] of Object.entries(retained))$(k).value=v;
  $('timingConfirmed').checked=confirmed;clockAnchor=anchor;setMode();updateGridStatus();
}
window.lockStructure=()=>{
  structurePanel.querySelectorAll('button,input,select').forEach(e=>{
    e.disabled=busy||!state?.analysis||(e===bookmarkImport&&!state?.reference_marks?.bookmarks?.length)||
      (e===kiaiImport&&!state?.reference_marks?.kiai_transitions?.length)||
      ([sectionSave,expressionAuto,expressionSave].includes(e)&&!state?.analysis?.structure);
  });
  const planned=!!state?.analysis?.structure;
  const ratioLabel=$('ln_ratio').closest('label');
  if(ratioLabel)ratioLabel.firstChild.textContent=planned?'持续声部倾向 ':'LN 比例目标 ';
  $('ln_ratio').disabled=busy||(planned&&state?.expression?.source==='manual');
  $('phrase_ln').disabled=busy||planned;
  $('phrase_ln').title=planned?'请在逐段声部方案里选择连续人声乐句':'';
};
window.renderStructure=()=>{
  structurePanel.classList.toggle('hidden',$('mode').value==='remap');
  structureRows.replaceChildren();expressionAudit.replaceChildren();expressionDraft=null;
  const structure=state?.analysis?.structure;
  structureInfo.textContent=structure?`${structure.segments.length} 段。${structure.notice||''} 保存声部方案后按段执行；自动方案会随全局偏好更新。${state?.expression?.notice||'LN 数值不强制凑比例。'}`:
    '先检测音乐变化，再决定每段采哪些声部。旧项目可以直接检测段落，无需重新运行音高模型。';
  structureBoundaries.value=structure?structure.boundaries.slice(1,-1).map(t=>t/1000).join(', '):'';
  if(state?.expression&&structure){
    expressionDraft=structuredClone(state.expression);
    for(const [index,section] of expressionDraft.sections.entries()){
      const row=document.createElement('details');row.className='structure-section';
      const summary=document.createElement('summary');summary.textContent=`${section.label||'段 '+(index+1)} · ${fmt(section.start_ms/1000)}–${fmt(section.end_ms/1000)} · ${section.similarity_group||''}`;
      const jump=document.createElement('button');jump.className='quiet';jump.textContent='定位试听';jump.onclick=()=>{seek(section.start_ms/1000);$('auditionStart').value=(section.start_ms/1000).toFixed(2);};
      const main=document.createElement('select');main.dataset.sectionMain=index;
      options(main,[['','不指定主声部（均衡取舍）'],...Object.keys(section.voices).map(v=>[v,sourceNames[v]||v])]);
      main.value=Object.keys(section.voices).find(v=>section.voices[v].role==='primary')||'';
      main.onchange=()=>{section.main_voice=main.value||null;for(const [v,p] of Object.entries(section.voices))p.role=v===main.value?'primary':'support';};
      const mainLabel=document.createElement('label');mainLabel.className='field';mainLabel.textContent='本段主声部';mainLabel.append(main);
      const choices=document.createElement('div');choices.className='structure-voices';
      for(const [voice,policy] of Object.entries(section.voices)){
        const voiceGroup=document.createElement('div');voiceGroup.className='structure-voice';
        const label=document.createElement('label');label.className='field';label.textContent=sourceNames[voice]||voice;
        const choice=document.createElement('select');choice.dataset.expressionVoice=voice;choice.dataset.expressionSection=index;
        options(choice,expressionOptions(voice));choice.value=policy.mode;
        choice.onchange=()=>policy.mode=choice.value;label.append(choice);voiceGroup.append(label);
        const activity=policy.activity;
        if(activity&&Number.isFinite(activity.local_attack_groups)){
          const detail=document.createElement('p');detail.className='muted compact';
          detail.dataset.expressionActivity=voice;
          detail.textContent=`本段 ${activity.local_attack_groups} 个候选音组；跨段延续 ${activity.carried_candidate_count} 音；可用持续区间覆盖 ${Math.round(activity.observed_interval_coverage*100)}%。`;
          if(activity.carried_candidate_count)detail.textContent+=' 延续音沿用起音所在段的表达。';
          if(activity.uncertain_tails)detail.textContent+=` ${activity.uncertain_tails} 个音尾未决。`;
          detail.title='这些是检测候选的活动统计，不表示声部重要性，也不证明模型音尾准确。';
          voiceGroup.append(detail);
        }
        const ids=[...new Set((state.analysis.candidates||[]).filter(n=>n.voice===voice&&n.instrument_id&&section.start_ms<=n.start&&n.start<section.end_ms).map(n=>n.instrument_id))].sort();
        if(ids.length){
          const controls=document.createElement('details');controls.className='instrument-modes';
          const caption=document.createElement('summary');caption.textContent=`按模型类别细分（${ids.length}）`;
          const hint=document.createElement('p');hint.className='muted compact';
          hint.textContent='类别是模型推测，并非独立声轨。可单独改变长短或启用本段未采的类别；主次沿用声部，声部总静音仍优先。';
          controls.append(caption,hint);
          for(const id of ids){
            const field=document.createElement('label');field.className='field';field.textContent=instrumentName(id);
            const select=document.createElement('select');select.dataset.expressionInstrument=id;select.dataset.instrumentVoice=voice;select.dataset.instrumentSection=index;
            options(select,[['inherit','继承声部方案'],...expressionOptions(voice)]);
            const modes=section.instrument_modes?.[voice]||{};
            select.value=Object.hasOwn(modes,id)?modes[id]:'inherit';
            select.onchange=()=>{section.instrument_modes={...section.instrument_modes,[voice]:{...section.instrument_modes?.[voice],[id]:select.value}};};
            field.append(select);controls.append(field);
          }
          voiceGroup.append(controls);
        }
        choices.append(voiceGroup);
      }
      const reasoning=document.createElement('p');reasoning.className='muted compact';
      reasoning.textContent=section.reason||'';
      row.append(summary,jump,mainLabel,choices,reasoning);structureRows.append(row);
    }
  }
  const decisions=variant()?.selection?.decisions;
  if(decisions){
    const title=document.createElement('p');title.className='muted compact';
    title.textContent='当前候选的省音记录（新方案需重新生成）：'+Object.entries(variant().selection.omission_reasons||{}).map(([k,v])=>`${omissionNames[k]||k} ${v}`).join('；');expressionAudit.append(title);
    const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='查看省略音的位置（前 80 个，全部记录在分析 JSON）';details.append(summary);
    for(const entry of decisions.filter(n=>n.decision==='omitted').slice(0,80)){
      const b=document.createElement('button');b.className='quiet';b.textContent=`${fmt(entry.start/1000)} ${sourceNames[entry.voice]||entry.voice}${entry.instrument_id?' / '+instrumentName(entry.instrument_id):''} · ${omissionNames[entry.reason]||entry.reason}`;
      b.onclick=()=>{setView('evidence');seek(Math.max(0,entry.start/1000-.4));};details.append(b);
    }
    expressionAudit.append(details);
  }
  window.lockStructure();
};
async function structureJob(payload){
  const retained=Object.fromEntries(['bpm','offset','division','audioShift','auditionOffset','mode'].map(k=>[k,$(k).value]));
  const confirmed=$('timingConfirmed').checked,anchor=clockAnchor;
  await runJob('structure',{...payload,settings:settings(),audio_shift_ms:clockShift()});
  for(const [k,v] of Object.entries(retained))$(k).value=v;
  $('timingConfirmed').checked=confirmed;clockAnchor=anchor;setMode();updateGridStatus();
}
sectionDetect.onclick=guarded(()=>structureJob({action:'detect'}));
bookmarkImport.onclick=guarded(()=>structureJob({action:'bookmarks'}));
kiaiImport.onclick=guarded(()=>structureJob({action:'kiai'}));
sectionSave.onclick=guarded(()=>{
  const text=structureBoundaries.value.trim();
  const boundaries=text?text.split(/[,，\s]+/).map(v=>Math.round(Number(v)*1000)):[];
  if(boundaries.some(v=>!Number.isFinite(v)))throw Error('分界请输入秒数，以逗号分隔');
  return structureJob({action:'edit',boundaries});
});
async function saveExpression(payload){
  if(busy||!state)return;setBusy(true);
  try{preserveTimingLoad(await api(`/api/projects/${state.id}/expression`,payload));msg('已保存逐段声部方案；重新生成后生效。');}
  finally{setBusy(false);}
}
expressionAuto.onclick=guarded(()=>saveExpression({action:'automatic',settings:settings()}));
expressionSave.onclick=guarded(()=>saveExpression({expression:expressionDraft}));
$('mode').addEventListener('change',()=>window.renderStructure());
window.renderStructure();
