'use strict';
let recommendationRequest=0, recommendationResult=null;
const recommendationPanel=document.createElement('details');
recommendationPanel.id='recommendationPanel';recommendationPanel.className='review-panel';recommendationPanel.open=true;
const recommendationSummary=document.createElement('summary');recommendationSummary.textContent='这首歌怎样写 · 本地编排建议';
const recommendationInfo=document.createElement('p');recommendationInfo.className='muted compact';
const recommendationChoices=document.createElement('div');recommendationChoices.id='recommendationChoices';
const recommendationRefresh=document.createElement('button');recommendationRefresh.id='recommendationRefresh';recommendationRefresh.className='quiet';recommendationRefresh.textContent='根据当前键数和 BPM 更新建议';
recommendationPanel.append(recommendationSummary,recommendationInfo,recommendationChoices,recommendationRefresh);
document.querySelector('article .project-head').after(recommendationPanel);

window.renderRecommendations=async()=>{
  const request=++recommendationRequest;
  recommendationResult=null;recommendationChoices.replaceChildren();
  recommendationPanel.classList.toggle('hidden',$('mode').value==='remap');
  const chosen=state?.plan?.source==='local-audio-recommendation'?state.plan.choice:null;
  recommendationSummary.textContent='这首歌怎样写 · '+(chosen?`已采用${({melody:'旋律与放手',hybrid:'均衡 Hybrid',layers:'密集多声部'})[chosen]||'本地方案'}`:'本地编排建议');
  recommendationRefresh.disabled=busy||!state?.analysis;
  if(!state?.analysis){recommendationInfo.textContent='分析音频后，比较旋律与放手、均衡 Hybrid、密集多声部三个方向。无需 API。';return;}
  recommendationInfo.textContent='正在根据本曲声音证据整理建议…';
  try{
    const result=await api(`/api/projects/${state.id}/recommendations`,{settings:settings(),bpm:Number($('bpm').value)});
    if(request!==recommendationRequest)return;
    recommendationResult=result;
    recommendationInfo.textContent=`${result.keys}K · ${result.summary.attacks_per_second.toFixed(1)} 次检测起音/秒 · ${(result.summary.sustain_fraction*100).toFixed(0)}% 候选有持续区间。${result.reference.available?`参考 ${result.reference.charts} 份谱、${result.reference.songs} 个歌曲分组，其中 LN≥50% 的有 ${result.reference.ln_charts} 份。`:'当前使用保守规则，尚未加载该键数的语料统计。'}${result.notice}`;
    for(const option of result.options){
      const card=document.createElement('div');card.className='recommendation-card';
      const heading=document.createElement('strong');heading.textContent=option.title+(result.suggested===option.id?' · 建议先试':'');
      const text=document.createElement('p');text.className='muted compact';text.textContent=`${option.description}。全局上限 ${option.preferences.density} 物件/秒，LN 目标 ${(option.preferences.ln_ratio*100).toFixed(0)}%。`;
      const details=document.createElement('details');const label=document.createElement('summary');label.textContent='查看分段理由';details.append(label);
      for(const section of option.section_reasons){
        const line=document.createElement('p');line.className='muted compact';
        line.textContent=`${fmt(section.start_ms/1000)}–${fmt(section.end_ms/1000)}：${section.design} · 上限 ${section.density}/秒`;
        if(section.opportunities?.length)line.textContent+='。可尝试：'+section.opportunities.map(o=>o.name).join('、');
        line.title=(section.opportunities||[]).map(o=>o.reason).join('\n');
        details.append(line);
      }
      const button=document.createElement('button');button.className='secondary';button.dataset.recommendation=option.id;button.textContent='采用这个方案';button.disabled=busy;
      button.onclick=guarded(async()=>{
        if(!recommendationResult)throw Error('请先更新建议');
        const retained=Object.fromEntries(['bpm','offset','division','audioShift','auditionOffset','mode','analysisEngine'].map(k=>[k,$(k).value]));
        const confirmed=$('timingConfirmed').checked,anchor=clockAnchor;
        setBusy(true);
        try{
          const next=await api(`/api/projects/${state.id}/apply-recommendation`,{choice:option.id,settings:settings(),bpm:Number($('bpm').value),evidence_signature:result.evidence_signature,context_signature:result.context_signature});
          loadState(next);
          for(const [key,value] of Object.entries(retained))$(key).value=value;
          $('timingConfirmed').checked=confirmed;clockAnchor=anchor;setMode();updateGridStatus();
          recommendationPanel.open=false;
          msg(`已采用「${option.title}」及分段建议。检查拍线后点击生成，可继续调整参数。`);
        }finally{setBusy(false);window.renderRecommendations();}
      });
      card.append(heading,text,details,button);recommendationChoices.append(card);
    }
  }catch(error){if(request===recommendationRequest)recommendationInfo.textContent=error.message;}
};
recommendationRefresh.onclick=guarded(()=>window.renderRecommendations());
document.querySelectorAll('[data-key]').forEach(button=>button.addEventListener('click',()=>window.renderRecommendations()));
$('bpm').addEventListener('change',()=>window.renderRecommendations());
$('mode').addEventListener('change',()=>recommendationPanel.classList.toggle('hidden',$('mode').value==='remap'));
for(const id of ['halfBpm','doubleBpm'])$(id).addEventListener('click',()=>window.renderRecommendations());
window.renderRecommendations();
